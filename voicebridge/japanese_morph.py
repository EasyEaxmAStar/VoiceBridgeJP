"""Resident CPU-only OpenJTalk frontend in the existing Python 3.11 environment."""
import contextlib, json, os, queue, subprocess, sys, threading
from pathlib import Path


class Morphology:
    def __init__(self):
        root=Path(__file__).resolve().parents[1]
        self._lock=threading.Lock();self._responses=queue.Queue(2)
        self.process=subprocess.Popen([str(root/'.venv-gptsovits/Scripts/python.exe'),'-X','utf8','-u','-m','voicebridge.japanese_morph'],
            cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self._reader=threading.Thread(target=self._read,daemon=True);self._reader.start()
        try:self('よろしくお願いします。')
        except BaseException:self.close();raise
    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line)>512*1024:raise ValueError('Oversized morphology response')
                self._responses.put(json.loads(line),timeout=1)
        except Exception:pass
        with contextlib.suppress(queue.Full):self._responses.put_nowait({'error':'Morphology worker exited'})
    def __call__(self,text):
        if not isinstance(text,str) or not 1<=len(text)<=20000:raise ValueError('Invalid Japanese text budget')
        with self._lock:
            if self.process.poll() is not None:raise RuntimeError('Japanese validation worker unavailable')
            self.process.stdin.write(json.dumps(text,ensure_ascii=False)+'\n');self.process.stdin.flush()
            try:result=self._responses.get(timeout=5)
            except queue.Empty:
                self.process.kill();raise TimeoutError('Japanese validation timed out; text is blocked')
            if not isinstance(result,list):raise RuntimeError('Japanese validation failed; text is blocked')
            return result
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        self._reader.join(timeout=1)
        if self.process.stdout:self.process.stdout.close()


def main():
    # Some optional frontend notices go to stdout at import. IPC remains JSON only.
    with contextlib.redirect_stdout(sys.stderr):import pyopenjtalk
    for line in sys.stdin:
        try:
            if len(line)>100000:raise ValueError('Text too large')
            text=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):result=pyopenjtalk.run_frontend(text)
        except Exception:result={'error':'Invalid morphology request'}
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
