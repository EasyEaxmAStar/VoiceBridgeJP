"""Qt signals around the existing controller's bounded anonymous-pipe interface."""
from __future__ import annotations
import json,os,queue,subprocess,threading,uuid
from pathlib import Path
from PySide6.QtCore import QObject,Signal
from .server_process import ProcessJob
from .release_voices import DEFAULT

ROOT=Path(__file__).resolve().parents[1]


class ControllerClient(QObject):
    message=Signal(dict)
    failed=Signal(str)
    exited=Signal(int)

    def __init__(self,*,preset=DEFAULT,connect=None,port=8765,fixture_input=False,parent=None):
        super().__init__(parent)
        self.preset,self.runtime_url,self.port=preset,connect,port
        self.fixture_input=fixture_input
        self.directory=ROOT/'.model-cache/gui-sessions'/uuid.uuid4().hex
        self.directory.mkdir(parents=True,exist_ok=False)
        self.commands=queue.Queue(8);self.process=None;self.job=None;self.closing=False
        self.done=threading.Event();self.pending=0;self.lock=threading.Lock()

    def start(self):
        self.thread=threading.Thread(target=self._run,name='controller-owner',daemon=True);self.thread.start()

    def send(self,command,**values):
        if self.closing:return False
        with self.lock:
            if self.pending>=8:
                self.failed.emit('控制命令队列已满，请等待当前操作完成');return False
            try:self.commands.put_nowait(dict(command=command,**values))
            except queue.Full:self.failed.emit('控制队列已满');return False
            self.pending+=1
        return True

    def _write(self):
        try:
            while self.process.poll() is None:
                try:value=self.commands.get(timeout=.2)
                except queue.Empty:continue
                self.process.stdin.write(json.dumps(value,ensure_ascii=False)+'\n');self.process.stdin.flush()
                if value['command']=='shutdown':return
        except (OSError,ValueError):pass

    def _run(self):
        code=-1
        try:
            gate=self.directory/'owned.gate'
            argv=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-u','-m','voicebridge.hot_cli',
                '--json-lines','--preset',self.preset,'--port',str(self.port),'--parent-gate',str(gate),
                '--report',str(self.directory/'controller-report.json')]
            if self.runtime_url:argv+=['--connect',self.runtime_url]
            if self.fixture_input:argv+=['--fixture-input']
            (self.directory/'launch.json').write_text(json.dumps(dict(command=argv,fixture_input=self.fixture_input),indent=2),encoding='utf8')
            self.job=ProcessJob()
            with (self.directory/'stderr.log').open('wb') as error,(self.directory/'events.jsonl').open('w',encoding='utf8') as journal:
                self.process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1'),stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=error,text=True,encoding='utf8',errors='strict',bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                self.job.assign(self.process);gate.touch(exist_ok=False)
                writer=threading.Thread(target=self._write,name='controller-commands',daemon=True);writer.start()
                for line in self.process.stdout:
                    # Private local journal: final transcripts/status, no raw mic recording.
                    journal.write(line);journal.flush()
                    try:value=json.loads(line)
                    except ValueError:continue
                    if not isinstance(value,dict):continue
                    if value.get('type') in {'command_done','error'}:
                        with self.lock:self.pending=max(0,self.pending-1)
                    self.message.emit(value)
                code=self.process.wait(timeout=5)
                writer.join(timeout=1)
        except Exception as exc:self.failed.emit(f'控制器失败：{exc}')
        finally:
            if self.job:self.job.close()
            if self.process:
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
                for handle in (self.process.stdin,self.process.stdout):
                    if handle:handle.close()
            self.done.set();self.exited.emit(code)

    def shutdown(self):
        if self.closing:return
        self.closing=True
        while True:
            try:self.commands.get_nowait()
            except queue.Empty:break
        self.commands.put_nowait(dict(command='shutdown'))
        def deadline():
            if not self.done.wait(20):
                if self.job:self.job.close()
        threading.Thread(target=deadline,name='controller-shutdown-deadline',daemon=True).start()
