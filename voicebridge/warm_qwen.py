"""One resident async Qwen owner; no output device and no idle inference."""
import asyncio,time
from pathlib import Path
from .qwen_translation import QwenTranslation
from .qwen_output import accept_qwen,load_fillers,filler_key,FillerSkipped

class WarmQwen:
    def __init__(self,folder,*,model=None):
        root=Path(__file__).resolve().parents[1];folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        self.worker=QwenTranslation(model or root/'.model-cache/Qwen3-4B-Instruct-2507',folder/'qwen.log')
        from .hybrid_translation import HybridTranslator
        self.scheduler=HybridTranslator(None,self.worker,None,mode='SEQUENTIAL_FALLBACK')
        self.state='QWEN_LOADING';self.metrics={};self.calls=0;self.pending=0;self.mode='OPUS_FAST';self.task=None
    async def start(self,snapshot,lease=None):
        started=time.perf_counter()
        try:
            await self.worker.start();self.state='QWEN_WARMING'
            self.metrics.update(verification_seconds=self.worker.verification_seconds,model_initialization_seconds=self.worker.ready['initialization_seconds'])
            # Coordinate startup GPU work with the same server lease used by
            # committed sessions. Never acquire this lease inside a session.
            async def warm():
                rows=[]
                for text in ['请确认会议时间。','如果数据库连接没有问题，我们可以在下午三点开始测试。']:
                    output=await self.worker(text,snapshot);accept_qwen(text,output,snapshot)
                    rows.append(dict(source=text,output=output,metrics=dict(self.worker.last_metrics)))
                return rows
            if lease:
                wait_started=time.perf_counter()
                await asyncio.wait_for(lease.acquire(),120)
                self.metrics['startup_gpu_wait_seconds']=time.perf_counter()-wait_started
                try:rows=await warm()
                finally:lease.release()
            else:rows=await warm()
            self.metrics.update(startup_seconds=time.perf_counter()-started,warmups=rows,audio_discarded=True)
            self.state='QWEN_READY'
        except asyncio.CancelledError:self.worker.close();raise
        except Exception as exc:self.state='QWEN_ERROR';self.metrics['error']=f'{type(exc).__name__}: {exc}';self.worker.close()
    def status(self):
        if self.state=='QWEN_READY' and (not self.worker.process or self.worker.process.poll() is not None):
            self.worker.close();self.state='QWEN_ERROR'
        return dict(state=self.state,mode=self.mode,model_loads=self.worker.loads,requests=self.calls,submitted_requests=self.worker.sequence,pending=self.pending,
            pid=self.worker.process.pid if self.worker.process else None,metrics=self.metrics)
    async def translate(self,source,snapshot):
        fillers=load_fillers()
        if filler_key(source) in fillers:raise FillerSkipped('FILLER_SKIPPED')
        if self.state!='QWEN_READY':raise RuntimeError('QWEN_TECHNICAL_FAILURE: warmed worker not ready')
        if not self.worker.process or self.worker.process.poll() is not None:
            self.worker.close();self.state='QWEN_ERROR';raise RuntimeError('QWEN_TECHNICAL_FAILURE: warmed worker unavailable')
        if self.pending>=2:raise RuntimeError('Bounded Qwen request queue full')
        self.pending+=1;self.calls+=1;started=time.perf_counter()
        try:
            text,audit=await self.scheduler.recover_final(source,snapshot,fillers=fillers)
            audit.update(backend='QWEN_FALLBACK',seconds=time.perf_counter()-started,worker=dict(self.worker.last_metrics))
            return text,audit
        except Exception as exc:
            if not self.worker.process or self.worker.process.poll() is not None:
                self.worker.close();self.state='QWEN_ERROR';self.metrics['last_error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:self.pending-=1
    async def close(self):
        if self.task and not self.task.done():self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        self.worker.close()
