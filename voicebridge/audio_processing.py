"""Reversible utterance processing; never used by the DIRECT microphone route."""
import hashlib,shutil,subprocess,time
import numpy as np
import asyncio


class TempoStream:
    """One stateful FFmpeg filter per utterance, with bounded pipes/backpressure."""
    def __init__(self,rate,speed,sink):
        if not .75<=speed<=1.5:raise ValueError('Invalid playback speed')
        self.rate,self.speed,self.sink=rate,speed,sink;self.process=None;self.reader=None;self.error_reader=None
        self.original=[];self.processed=[];self.raw_samples=0;self.output_samples=0;self.first=None
        self.started=time.perf_counter();self.finished=False;self.command=None
        self.emit_trace=[]
    async def start(self):
        if self.speed==1.:return
        executable=shutil.which('ffmpeg')
        if not executable:raise RuntimeError('FFmpeg is required for the selected streaming tempo stage')
        self.command=[executable,'-nostdin','-hide_banner','-loglevel','error','-probesize','32','-analyzeduration','0',
            '-f','f32le','-ar',str(self.rate),'-ac','1','-i','pipe:0','-af',f'atempo={self.speed}',
            '-f','f32le','-flush_packets','1','-ar',str(self.rate),'-ac','1','pipe:1']
        self.process=await asyncio.create_subprocess_exec(*self.command,stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,limit=65536,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.error_reader=asyncio.create_task(self.process.stderr.read(8192))
        self.reader=asyncio.create_task(self._read())
    async def _emit(self,x):
        if not np.isfinite(x).all() or np.max(np.abs(x),initial=0)>1.0001:raise RuntimeError('Invalid processed PCM')
        self.output_samples+=len(x)
        if self.output_samples>self.rate*170:raise RuntimeError('Bounded processed-audio budget exceeded')
        if len(x):
            if self.first is None:self.first=time.perf_counter()-self.started
            self.emit_trace.append(dict(seconds=time.perf_counter()-self.started,samples=len(x)))
            if len(self.emit_trace)>128:del self.emit_trace[0]
            self.processed.append(x.copy());await self.sink(x)
    async def _read(self):
        pending=b''
        while True:
            raw=await self.process.stdout.read(round(self.rate*.02)*4)
            if not raw:break
            pending+=raw;length=len(pending)//4*4
            if length:await self._emit(np.frombuffer(pending[:length],dtype='<f4').copy());pending=pending[length:]
        if pending:raise RuntimeError('Truncated float PCM at processor EOF')
    async def add(self,x):
        if self.finished:raise RuntimeError('Processor was finalized')
        x=np.asarray(x,dtype='float32')
        if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Invalid processing input')
        self.raw_samples+=len(x)
        if self.raw_samples>self.rate*120:raise RuntimeError('Bounded input audio budget exceeded')
        self.original.append(x.copy())
        if self.speed==1.:await self._emit(x.copy());return
        if self.reader.done():self.reader.result()
        self.process.stdin.write(x.astype('<f4',copy=False).tobytes());await self.process.stdin.drain()
    async def finish(self):
        if self.finished:raise RuntimeError('Duplicate processor completion')
        self.finished=True
        if self.process:
            self.process.stdin.close()
            await asyncio.wait_for(self.reader,30)
            code=await asyncio.wait_for(self.process.wait(),3);error=await self.error_reader
            if code:raise RuntimeError('FFmpeg processing failed: '+error.decode('utf8',errors='replace'))
        if not self.output_samples:raise RuntimeError('No playable processing PCM')
        a=np.concatenate(self.original);b=np.concatenate(self.processed)
        return a,b,dict(method='ffmpeg_atempo_stream' if self.speed!=1 else 'bypass',speed=self.speed,rate=self.rate,
            original_samples=len(a),processed_samples=len(b),original_float_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            processed_float_sha256=hashlib.sha256(b.tobytes()).hexdigest(),seconds=time.perf_counter()-self.started,
            first_playable_processing_seconds=self.first,command=self.command,filter_instances=1,
            chunk_trace=self.emit_trace,
            reversible=True,browser_equivalence='NOT_VERIFIED',buffer_limit_seconds=120)
    async def cancel(self):
        if self.process and self.process.returncode is None:
            try:self.process.terminate()
            except ProcessLookupError:pass
            try:await asyncio.wait_for(self.process.wait(),2)
            except asyncio.TimeoutError:self.process.kill();await self.process.wait()
        for task in [self.reader,self.error_reader]:
            if task and not task.done():task.cancel()
        await asyncio.gather(*(t for t in [self.reader,self.error_reader] if t),return_exceptions=True)


def pitch_preserving(samples,rate=24000,speed=1.25):
    x=np.asarray(samples,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.max(np.abs(x))>1.0001:raise ValueError('Invalid source PCM')
    if not .75<=speed<=1.5:raise ValueError('Playback speed must be .75–1.50')
    before=hashlib.sha256(x.tobytes()).hexdigest();t=time.perf_counter()
    if speed==1.:y=x.copy();command=None
    else:
        executable=shutil.which('ffmpeg')
        if not executable:raise RuntimeError('FFmpeg is required for the explicitly selected pitch-preserving stage')
        command=[executable,'-nostdin','-hide_banner','-loglevel','error','-f','f32le','-ar',str(rate),'-ac','1',
            '-i','pipe:0','-af',f'atempo={speed}','-f','f32le','-ac','1','-ar',str(rate),'pipe:1']
        result=subprocess.run(command,input=x.astype('<f4').tobytes(),capture_output=True,timeout=30,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise RuntimeError(result.stderr.decode('utf8',errors='replace')[-2000:])
        if len(result.stdout)%4:raise RuntimeError('Malformed FFmpeg float PCM payload')
        y=np.frombuffer(result.stdout,dtype='<f4').copy()
    if not len(y) or not np.isfinite(y).all() or np.max(np.abs(y))>1.0001:raise RuntimeError('Invalid processed PCM')
    if hashlib.sha256(x.tobytes()).hexdigest()!=before:raise RuntimeError('Source PCM was modified')
    return y,dict(method='ffmpeg_atempo' if speed!=1. else 'bypass',speed=speed,rate=rate,
        original_samples=len(x),processed_samples=len(y),original_float_sha256=before,
        processed_float_sha256=hashlib.sha256(y.tobytes()).hexdigest(),seconds=time.perf_counter()-t,
        command=command,reversible=True,browser_equivalence='NOT_VERIFIED',
        limitation='Whole utterance buffering adds latency; native source preserved, output can be regenerated with processing disabled.')
