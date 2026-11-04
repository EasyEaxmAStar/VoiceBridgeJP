"""Owned video subprocess with latest-control exchange; no video dependencies."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import subprocess
import threading
import time
from pathlib import Path

from .camera_events import CameraCommand
from .server_process import ProcessJob


class CameraVideoProcess:
    def __init__(self, config_path, *, python=None):
        root=Path(__file__).resolve().parents[1]
        python=Path(python or root/'.venv-camera/Scripts/python.exe')
        if not python.is_file():raise RuntimeError('Video environment missing: install requirements-camera.lock.txt into .venv-camera')
        self.latest=deque(maxlen=1);self.stop=threading.Event();self.error=None;self.last_status={}
        self.job=ProcessJob();self.process=None
        try:
            self.process=subprocess.Popen([str(python),'-X','utf8','-u','-m','voicebridge.camera_worker','--controlled','--config',str(Path(config_path).resolve())],
                cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            self.job.assign(self.process)
        except Exception:
            if self.process:self.process.kill();self.process.wait(timeout=2)
            self.job.close();raise
        self.writer=threading.Thread(target=self._write,name='camera-video-control',daemon=True)
        self.reader=threading.Thread(target=self._read,name='camera-video-status',daemon=True)
        self.writer.start();self.reader.start()

    def publish(self,snapshot):
        self.latest.append(asdict(CameraCommand.from_snapshot(snapshot)))

    def _write(self):
        current=None
        try:
            while not self.stop.wait(.02):
                try:current=self.latest.pop()
                except IndexError:continue
                self.process.stdin.write((json.dumps(current,separators=(',',':'))+'\n').encode());self.process.stdin.flush()
            # Actor already publishes its OFF fence before close. Flush the
            # newest fence, then shutdown. No blocking write on the actor.
            if self.latest:
                self.process.stdin.write((json.dumps(self.latest.pop())+'\n').encode())
            self.process.stdin.write(b'{"shutdown":true}\n');self.process.stdin.flush()
        except (OSError,ValueError):self.error='Video worker control disconnected'

    def _read(self):
        try:
            while True:
                line=self.process.stdout.readline(8193)
                if not line:break
                if len(line)>8192:raise ValueError('Oversized camera status')
                try:data=json.loads(line)
                except ValueError:continue  # native driver messages are not commands
                self.last_status=data
                if data.get('type')=='camera_error':self.error=data.get('error','Video worker failed')
        except Exception as exc:self.error=type(exc).__name__
        finally:
            if not self.stop.is_set() and not self.error:self.error='Video worker output closed; check camera configuration and backend availability'

    def status(self):
        return dict(pid=self.process.pid,alive=self.process.poll() is None,error=self.error,last=self.last_status)

    def close(self):
        self.stop.set()
        try:self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:self.job.close();self.process.wait(timeout=2)
        finally:
            self.job.close()
            for handle in (self.process.stdin,self.process.stdout):
                try:handle.close()
                except (OSError,ValueError):pass
            self.writer.join(1);self.reader.join(1)
