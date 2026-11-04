"""Own one native localhost model server and its child process tree."""
import asyncio
from contextlib import suppress
import ctypes
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid


class ProcessJob:
    """Windows kill-on-close Job Object; never contains unrelated processes."""
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Managed model server requires native Windows")
        class Basic(ctypes.Structure):
            _fields_ = [("process_time",ctypes.c_int64),("job_time",ctypes.c_int64),("flags",W.DWORD),
                ("min_ws",ctypes.c_size_t),("max_ws",ctypes.c_size_t),("active",W.DWORD),
                ("affinity",ctypes.c_size_t),("priority",W.DWORD),("scheduling",W.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name,ctypes.c_uint64) for name in ("ro","wo","oo","rt","wt","ot")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic",Basic),("io",IO),("pm",ctypes.c_size_t),("jm",ctypes.c_size_t),
                        ("peak_p",ctypes.c_size_t),("peak_j",ctypes.c_size_t)]
        api = self.api = ctypes.WinDLL("kernel32",use_last_error=True)
        api.CreateJobObjectW.argtypes = [ctypes.c_void_p,W.LPCWSTR]
        api.CreateJobObjectW.restype = W.HANDLE
        api.SetInformationJobObject.argtypes = [W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD]
        api.AssignProcessToJobObject.argtypes = [W.HANDLE,W.HANDLE]
        api.CloseHandle.argtypes = [W.HANDLE]
        self.handle = api.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle,W.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class ManagedServer:
    def __init__(self, reference, *, port=8765, root=None, startup_timeout=180,
                 module='voicebridge.production_server', environment=None):
        self.reference = reference
        self.port = port
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.startup_timeout = startup_timeout
        self.module,self.environment=module,dict(environment or {})
        self.process = self.job = self._lock = self._log = None
        self.token = uuid.uuid4().hex
        self.url = f"ws://127.0.0.1:{port}/v1/realtime/translations"

    def _launch(self):
        import msvcrt
        cache = self.root/".model-cache/live-server"
        cache.mkdir(parents=True,exist_ok=True)
        self._lock = (cache/f"port-{self.port}.lock").open("a+b")
        self._lock.seek(0,2)
        if not self._lock.tell():
            self._lock.write(b"\0")
            self._lock.flush()
        self._lock.seek(0)
        try:
            msvcrt.locking(self._lock.fileno(),msvcrt.LK_NBLCK,1)
            with socket.socket() as test:
                test.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
                test.bind(("127.0.0.1",self.port))
            self.job = ProcessJob()
            self.log_path = cache/f"server-{self.token}.log"
            self.gate_path = cache/f"server-{self.token}.gate"
            self._log = self.log_path.open("wb")
            env = dict(os.environ,ORT_GPTSOVITS_REF_AUDIO=self.reference.path,
                ORT_GPTSOVITS_REF_TEXT=self.reference.text,ORT_GPTSOVITS_REF_LANGUAGE=self.reference.language,
                HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",PYTHONUTF8="1")
            env.update(self.environment)
            self.process = subprocess.Popen([sys.executable,"-X","utf8","-u","-m",self.module,
                "--port",str(self.port),"--instance",self.token,"--gate",str(self.gate_path)],cwd=self.root,env=env,
                stdout=self._log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            self.job.assign(self.process)
            self.gate_path.touch(exist_ok=False)
        except BaseException:
            self.close()
            raise

    async def start(self):
        self.started_at = time.perf_counter()
        await asyncio.to_thread(self._launch)
        import urllib.request
        def ready():
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{self.port}/health",timeout=.3) as response:
                    return json.load(response).get("voicebridge_instance") == self.token
            except OSError:
                return False
        deadline = time.monotonic()+self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Production server exited; inspect {self.log_path}")
            if await asyncio.to_thread(ready):
                self.ready_at = time.perf_counter()
                self.startup_seconds = self.ready_at-self.started_at
                return
            await asyncio.sleep(.1)
        raise TimeoutError(f"Model server startup timed out; inspect {self.log_path}")

    def close(self):
        # Close the owned listening socket normally before killing the job.
        # On Windows, abrupt teardown with live HTTP sessions left our exclusive
        # port probe unable to bind during explicit worker-failure recovery.
        if self.module=='voicebridge.hot_server' and self.process and self.process.poll() is None:
            with suppress(OSError,subprocess.TimeoutExpired):
                import urllib.request
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                request=urllib.request.Request(f'http://127.0.0.1:{self.port}/runtime/shutdown',
                    data=json.dumps({'instance':self.token}).encode(),headers={'Content-Type':'application/json'})
                with opener.open(request,timeout=1) as response:response.read(1024)
                self.process.wait(timeout=4)
        if self.job:
            self.job.close()  # Also kills the resident GPT-SoVITS child.
            self.job = None
        if self.process:
            if self.process.poll() is None:
                with suppress(ProcessLookupError,PermissionError):
                    self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for handle in (self._log,self._lock):
            if handle:
                handle.close()
        self._log = self._lock = None
        gate = getattr(self,"gate_path",None)
        if gate and gate.exists():
            gate.unlink()
