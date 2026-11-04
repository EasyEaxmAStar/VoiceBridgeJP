"""Bounded resident experimental worker. Timeout/crash never yields fallback text."""
import asyncio,json,os,subprocess,sys,time,hashlib
from pathlib import Path
from .server_process import ProcessJob

class QwenTranslation:
    def __init__(self,path,log,*,timeout=8.,origin_path=None):
        self.path=Path(path).resolve();self.log=Path(log);self.timeout=timeout;self.lock=asyncio.Lock()
        self.origin_path=Path(origin_path) if origin_path else Path(__file__).resolve().parents[1]/'.model-cache/hybrid-translation/model-origin.json'
        self.process=None;self.job=None;self.sequence=0;self.last_metrics={};self.loads=0
    async def start(self):
        if self.process:raise RuntimeError('Worker already started')
        def verify():
            origin=json.loads(self.origin_path.read_text(encoding='utf8'))
            if origin['repository']!='Qwen/Qwen3-4B-Instruct-2507':raise ValueError('Unexpected translation model identity')
            for info in origin['files']:
                target=self.path/info['rfilename']
                if not target.is_file() or target.stat().st_size!=info['size']:raise FileNotFoundError('Incomplete pinned Qwen weights: '+target.name)
                with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if digest!=info['lfs']['sha256']:raise ValueError('Qwen artifact SHA mismatch')
            return origin['revision']
        verify_started=time.perf_counter();self.revision=await asyncio.to_thread(verify)
        self.verification_seconds=time.perf_counter()-verify_started
        self.error=self.log.open('xb');self.job=ProcessJob()
        self.process=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','voicebridge.qwen_translation_worker',str(self.path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.error,text=True,encoding='utf8',
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),creationflags=subprocess.CREATE_NO_WINDOW)
        self.job.assign(self.process)
        try:
            line=await asyncio.wait_for(asyncio.to_thread(self.process.stdout.readline),120)
            self.ready=json.loads(line);assert self.ready['ready'];self.loads+=1
        except BaseException:self.close();raise
    async def __call__(self,source,snapshot):
        parent_started=time.perf_counter()
        async with self.lock:
            queue_wait=time.perf_counter()-parent_started
            if not self.process or self.process.poll() is not None:raise RuntimeError('Qwen worker unavailable')
            self.sequence+=1;identifier=self.sequence
            read=None
            request=dict(id=identifier,source=source,terms=[dict(source=r['matched'],target=r['target_ja']) for r in snapshot['dictionary'].source_matches(source)])
            request['repair']=bool(snapshot.get('translation_repair',False))
            request['prompt_style']=snapshot.get('qwen_prompt_style','expanded')
            request['measure_breakdown']=snapshot.get('qwen_measure_breakdown',False) is True
            request['max_new_tokens']=snapshot.get('qwen_token_budget',160)
            if type(request['max_new_tokens']) is not int or not 64<=request['max_new_tokens']<=256:raise ValueError('Invalid bounded Qwen token budget')
            try:
                self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');self.process.stdin.flush()
                read=asyncio.create_task(asyncio.to_thread(self.process.stdout.readline))
                line=await asyncio.wait_for(asyncio.shield(read),self.timeout)
                if len(line)>65536:raise ValueError('Qwen response oversized')
                result=json.loads(line)
                result.update(parent_request_seconds=time.perf_counter()-parent_started,queue_wait_seconds=queue_wait)
                if result['id']!=identifier:raise ValueError('Qwen sequence mismatch')
                self.last_metrics=result
                if result.get('cancelled'):raise RuntimeError('Worker cancelled current request')
                if result.get('truncated'):raise ValueError('Qwen output ended without EOS; refusing potentially incomplete text')
                self.last_metrics=result;return result['text']
            except (asyncio.CancelledError,TimeoutError):
                try:
                    self.process.stdin.write(json.dumps(dict(cancel=identifier))+'\n');self.process.stdin.flush()
                    if read:await asyncio.wait_for(asyncio.shield(read),2)
                except BaseException:self.close()
                raise
            except BaseException:
                # Hard cancellation is honest: terminate GPU work, invalidate
                # residency, require explicit restart. Never leave a read or
                # generation alive that could be confused with the next job.
                self.close();raise
    def close(self):
        if self.job:self.job.close();self.job=None
        if self.process:
            if self.process.poll() is None:self.process.kill()
            self.process.wait(timeout=5)
            for handle in [self.process.stdin,self.process.stdout]:
                if handle:handle.close()
            self.process=None
        if getattr(self,'error',None):self.error.close()
