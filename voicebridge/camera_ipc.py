"""Optional bounded ASR metadata pipe. Windows only; no audio/model data.

Uses CPython's native overlapped PipeConnection on a pipe created with an
explicit current-user-only DACL. Authentication secret is inherited only by
the owned server, never published in GUI status or recordings.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as W
import hmac
import json
import os
import queue
import secrets
import threading
import time

MAX_FRAME = 2048


def _secure_handle(address):
    import _winapi
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    adv = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.LocalFree.argtypes = [W.HLOCAL]
    adv.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, ctypes.POINTER(W.HANDLE)]
    adv.GetTokenInformation.argtypes = [W.HANDLE, ctypes.c_int, W.LPVOID, W.DWORD, ctypes.POINTER(W.DWORD)]
    adv.ConvertSidToStringSidW.argtypes = [W.LPVOID, ctypes.POINTER(W.LPWSTR)]
    adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [W.LPCWSTR, W.DWORD, ctypes.POINTER(W.LPVOID), ctypes.POINTER(W.DWORD)]
    token = W.HANDLE();size = W.DWORD();sid_text = W.LPWSTR();sd = W.LPVOID()
    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', W.DWORD), ('descriptor', W.LPVOID), ('inherit', W.BOOL)]
    kernel.CreateNamedPipeW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, W.DWORD, W.DWORD, W.DWORD, W.DWORD, ctypes.POINTER(SecurityAttributes)]
    kernel.CreateNamedPipeW.restype = W.HANDLE
    try:
        if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not adv.GetTokenInformation(token, 1, buf, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid = ctypes.cast(buf, ctypes.POINTER(W.LPVOID))[0]
        if not adv.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        sddl = f'D:P(A;;GA;;;{sid_text.value})'
        if not adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(sd), None):
            raise ctypes.WinError(ctypes.get_last_error())
        attrs = SecurityAttributes(ctypes.sizeof(SecurityAttributes), sd, False)
        handle = kernel.CreateNamedPipeW(address, 0x40000000 | 0x00080000 | 3,
            4 | 2 | 8, 1, MAX_FRAME, MAX_FRAME, 0, ctypes.byref(attrs))
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(handle), sddl
    finally:
        if token:kernel.CloseHandle(token)
        if sid_text:kernel.LocalFree(ctypes.cast(sid_text, W.HLOCAL))
        if sd:kernel.LocalFree(sd)


class PipeReader:
    """One owner-authenticated publisher. Receiver never runs application code."""
    def __init__(self, capacity=128):
        if os.name != 'nt':raise RuntimeError('Camera telemetry requires Windows')
        self.address = r'\\.\pipe\voicebridge-camera-' + secrets.token_hex(16)
        self.secret = secrets.token_hex(32);self.owner = secrets.token_hex(16)
        self.messages = queue.Queue(capacity);self.lost = threading.Event()
        self.stop = threading.Event();self.connected = threading.Event()
        self._handle, self.sddl = _secure_handle(self.address)
        self._connection = None;self.error = None
        self.thread = threading.Thread(target=self._read, name='camera-pipe-reader', daemon=True)
        self.thread.start()

    @property
    def environment(self):
        return dict(VOICEBRIDGE_CAMERA_PIPE=self.address, VOICEBRIDGE_CAMERA_SECRET=self.secret,
                    VOICEBRIDGE_CAMERA_OWNER=self.owner)

    def _read(self):
        import _winapi
        from multiprocessing.connection import PipeConnection
        connection = None;overlapped = None
        try:
            overlapped = _winapi.ConnectNamedPipe(self._handle, overlapped=True)
            while not self.stop.is_set():
                if _winapi.WaitForSingleObject(overlapped.event, 50) == 0:break
            if self.stop.is_set():
                overlapped.cancel();return
            overlapped.GetOverlappedResult(True)
            connection = self._connection = PipeConnection(self._handle)
            self._handle = None
            seq = 0;last = time.perf_counter();peer_at = -float('inf')
            while not self.stop.is_set():
                if not connection.poll(.05):
                    if time.perf_counter() - last > .5:raise TimeoutError('ASR telemetry heartbeat expired')
                    continue
                raw = connection.recv_bytes(MAX_FRAME)
                data = json.loads(raw)
                if (not isinstance(data, dict) or not isinstance(data.get('secret'), str)
                        or not hmac.compare_digest(data['secret'], self.secret)
                        or data.get('owner') != self.owner):
                    raise ValueError('Camera telemetry authentication failed')
                if type(data.get('seq')) is not int or data['seq'] != seq + 1:
                    raise ValueError('Camera telemetry sequence loss/replay')
                now = time.perf_counter();stamp = data.get('wire_at')
                if type(stamp) not in {int, float} or not peer_at <= stamp <= now or now-stamp > .5:
                    raise ValueError('Camera telemetry clock or delivery invalid')
                seq = data['seq'];peer_at = stamp;last = now
                data.pop('secret');self.connected.set()
                if data.get('kind') == 'LOST':raise ValueError('Publisher overflow/loss')
                if data.get('kind') not in {'HEARTBEAT', 'ASR_STARTED', 'JOB_FAILED', 'ASR_CANCELLED'}:
                    raise ValueError('Unknown camera telemetry kind')
                self.messages.put_nowait(data)
        except Exception as exc:
            if not self.stop.is_set():self.error = type(exc).__name__;self.lost.set()
        finally:
            if connection:connection.close()
            elif self._handle is not None:_winapi.CloseHandle(self._handle);self._handle = None

    def close(self):
        self.stop.set()
        self.thread.join(1)
        if self.thread.is_alive():raise RuntimeError('Camera pipe reader did not stop')


class ASRPublisher:
    def __init__(self, environment=None, capacity=128):
        env = os.environ if environment is None else environment
        self.address = env['VOICEBRIDGE_CAMERA_PIPE'];self.secret = env['VOICEBRIDGE_CAMERA_SECRET']
        self.owner = env['VOICEBRIDGE_CAMERA_OWNER']
        self.queue = queue.Queue(capacity);self.stop = threading.Event();self.failed = threading.Event()
        self._lock = threading.Lock();self._seq = 0;self.error = None
        self.thread = threading.Thread(target=self._write, name='camera-asr-publisher', daemon=True)
        self.thread.start()

    def publish(self, kind, context, at):
        """Inference boundary: bounded nonblocking scalar metadata admission."""
        if self.stop.is_set() or self.failed.is_set():return False
        try:
            self.queue.put_nowait(dict(kind=kind, session=context.session_id, epoch=context.epoch,
                                       job=context.utterance_id, at=at))
            return True
        except queue.Full:self.failed.set();return False

    def _write(self):
        import _winapi
        from multiprocessing.connection import PipeConnection
        connection = None
        try:
            deadline = time.monotonic() + 2
            while not self.stop.is_set():
                try:
                    handle = _winapi.CreateFile(self.address, 0xC0000000, 0, 0, 3, 0x40000000, 0)
                    _winapi.SetNamedPipeHandleState(handle, 2, None, None)
                    connection = PipeConnection(handle);break
                except OSError:
                    if time.monotonic() >= deadline:raise TimeoutError('Camera pipe unavailable')
                    self.stop.wait(.02)
            while connection and not self.stop.is_set():
                try:data = self.queue.get(timeout=.1)
                except queue.Empty:data = dict(kind='HEARTBEAT', at=time.perf_counter())
                if self.failed.is_set():data = dict(kind='LOST', at=time.perf_counter())
                self._seq += 1
                data.update(secret=self.secret, owner=self.owner, seq=self._seq)
                data['wire_at'] = time.perf_counter()
                raw = json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf8')
                if len(raw) > MAX_FRAME:raise ValueError('Oversized camera telemetry frame')
                # Overlapped bounded write: a stalled reader cannot strand an
                # inference worker or leave an uncancellable writer thread.
                ov, err = _winapi.WriteFile(connection.fileno(), raw, overlapped=True)
                deadline = time.monotonic() + .5
                while _winapi.WaitForSingleObject(ov.event, 50) != 0:
                    if self.stop.is_set() or self.failed.is_set() or time.monotonic() >= deadline:
                        self.failed.set()
                        ov.cancel();ov.GetOverlappedResult(True);return
                count, err = ov.GetOverlappedResult(True)
                if err or count != len(raw):raise OSError('Incomplete metadata write')
                if data['kind'] == 'LOST':return
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}';self.failed.set()
        finally:
            if connection:connection.close()

    def close(self):
        self.stop.set();self.thread.join(1)


def publish_optional(pipeline, kind, context, at):
    """Side-channel failure must never change the ASR/TTS application's result."""
    publisher = getattr(pipeline, 'voicebridge_camera_publisher', None)
    if publisher is not None and context is not None:
        try:publisher.publish(kind, context, at)
        except Exception:
            try:publisher.failed.set()
            except Exception:pass
