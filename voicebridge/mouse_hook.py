"""Opt-in Windows non-suppressing button-down listener. No movement/key telemetry."""
import ctypes,os,threading,queue,time
from ctypes import wintypes as W

class GlobalMouseHook:
    def __init__(self,callback,*,ignored_pid=None):
        self.callback=callback;self.ignored_pid=os.getpid() if ignored_pid is None else ignored_pid
        self.events=queue.Queue(16);self.stop_event=threading.Event();self.ready=threading.Event();self.error=None;self.thread=None;self.thread_id=None
        self.dropped=0
    @property
    def active(self):return bool(self.thread and self.thread.is_alive() and self.ready.is_set() and not self.error)
    def start(self):
        if self.active:return
        if os.name!='nt':raise RuntimeError('GLOBAL_MOUSE requires native Windows')
        self.thread=threading.Thread(target=self._run,name='global-mouse-listener',daemon=True);self.thread.start()
        if not self.ready.wait(2):self.close();raise TimeoutError('Mouse hook startup')
        if self.error:self.close();raise RuntimeError(self.error)
    def _run(self):
        user=ctypes.WinDLL('user32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        proc_type=ctypes.WINFUNCTYPE(ctypes.c_ssize_t,ctypes.c_int,W.WPARAM,W.LPARAM)
        user.SetWindowsHookExW.argtypes=[ctypes.c_int,proc_type,W.HINSTANCE,W.DWORD];user.SetWindowsHookExW.restype=W.HANDLE
        user.CallNextHookEx.argtypes=[W.HANDLE,ctypes.c_int,W.WPARAM,W.LPARAM];user.CallNextHookEx.restype=ctypes.c_ssize_t
        user.GetForegroundWindow.restype=W.HWND;user.GetWindowThreadProcessId.argtypes=[W.HWND,ctypes.POINTER(W.DWORD)]
        user.UnhookWindowsHookEx.argtypes=[W.HANDLE];user.PeekMessageW.argtypes=[ctypes.POINTER(W.MSG),W.HWND,W.UINT,W.UINT,W.UINT]
        self.thread_id=kernel.GetCurrentThreadId();hook=None;presses={}
        class MouseEvent(ctypes.Structure):
            _fields_=[('pt',W.POINT),('mouseData',W.DWORD),('flags',W.DWORD),('time',W.DWORD),('extra',ctypes.c_size_t)]
        @proc_type
        def receive(code,message,data):
            if code>=0 and message in (0x0201,0x0202,0x0204,0x0205):
                button='LEFT' if message in (0x0201,0x0202) else 'RIGHT'
                pid=W.DWORD();user.GetWindowThreadProcessId(user.GetForegroundWindow(),ctypes.byref(pid))
                point=ctypes.cast(data,ctypes.POINTER(MouseEvent)).contents.pt
                if message in (0x0201,0x0204):
                    # Only transient down/up coordinates to reject drags. Never
                    # listen to/store movement, coordinates or foreground names.
                    presses[button]=(point.x,point.y,pid.value,time.perf_counter())
                else:
                    down=presses.pop(button,None)
                    if down and pid.value==down[2] and pid.value!=self.ignored_pid and max(abs(point.x-down[0]),abs(point.y-down[1]))<=4 and time.perf_counter()-down[3]<=1:
                        try:self.events.put_nowait((button,time.perf_counter()))
                        except queue.Full:self.dropped+=1
            # Always pass the original click onward, including menus and drags.
            return user.CallNextHookEx(hook,code,message,data)
        try:
            hook=user.SetWindowsHookExW(14,receive,None,0)
            if not hook:raise ctypes.WinError(ctypes.get_last_error())
            self.ready.set();message=W.MSG()
            while not self.stop_event.is_set():
                while user.PeekMessageW(ctypes.byref(message),None,0,0,1):
                    user.TranslateMessage(ctypes.byref(message));user.DispatchMessageW(ctypes.byref(message))
                while not self.events.empty():
                    button,stamp=self.events.get_nowait();self.callback(button,stamp)
                self.stop_event.wait(.005)
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}';self.ready.set()
        finally:
            if hook:user.UnhookWindowsHookEx(hook)
    def close(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():self.thread.join(2)
        if self.thread and self.thread.is_alive():raise RuntimeError('Mouse listener did not stop')
