"""Accurate independent CABLE verification using actual WASAPI packets/QPC.

The retained PortAudio recorder remains available for comparison. This capture
worker reads actual GetBuffer packets, never clock-estimated silence or source PCM.
"""
import asyncio,json,subprocess,tempfile,time
from pathlib import Path
import soundfile as sf
from ort.config import Config
from .live import select_device
from .live_loopback import LoopbackRecorder


class NativeLoopbackRecorder(LoopbackRecorder):
    async def start(self):
        self.device=select_device(self.devices(),self.selection,'loopback')
        self.rate=int(self.device.default_samplerate);self.channels=2
        if self.rate!=48000 or self.device.max_input_channels<2:raise ValueError('Native packet verifier requires the selected48k stereo CABLE endpoint')
        root=Path(__file__).resolve().parents[1];directory=root/'.model-cache/native-cable-captures';directory.mkdir(exist_ok=True)
        self.directory=Path(tempfile.mkdtemp(prefix='capture-',dir=directory));self.output=self.directory/'capture.wav'
        self.log_path=self.directory/'capture.log';self._log=self.log_path.open('wb');self._stream=True
        command=[Config().gptsovits_python,'-X','utf8',str(root/'tools/native_cable_capture.py'),
            '--output',str(self.output),'--seconds',str(self.max_seconds),'--device-name',self.device.name]
        try:
            self.process=subprocess.Popen(command,cwd=root,stdout=self._log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            deadline=time.monotonic()+10
            while not self.output.with_suffix('.ready').exists():
                if self.process.poll() is not None:raise RuntimeError('Native verifier failed: '+str(self.log_path))
                if time.monotonic()>deadline:raise TimeoutError('Native WASAPI verifier startup')
                await asyncio.sleep(.01)
        except BaseException:
            await self.stop();self._log.close();raise
    async def stop(self):
        process=getattr(self,'process',None)
        if process:
            if process.poll() is None:
                self.output.with_suffix('.stop').touch(exist_ok=True)
                try:await asyncio.to_thread(process.wait,5)
                except subprocess.TimeoutExpired:process.terminate();await asyncio.to_thread(process.wait,3)
            self._log.close();self._stream=None
            if process.returncode: self.error='Native capture exited: '+str(self.log_path);return
            metadata=json.loads(self.output.with_suffix('.json').read_text(encoding='utf8'));self.native_metadata=metadata
            audio,rate=sf.read(self.output,dtype='float32',always_2d=True);self.blocks=[audio]
            if not len(audio) or rate!=48000:raise RuntimeError('Native recording is empty or changed format')
            self.clock_trace=[dict(sample=p['sample'],adc_mapped_at=p['qpc_seconds'],raw_adc=p['qpc_seconds'],received_at=p['received_at']) for p in metadata['packets']]
            self.origin=self.clock_trace[0]['adc_mapped_at'];self.original_adc=self.origin
            if metadata['discontinuities'] or metadata['position_gaps'] or metadata['timestamp_error_packets']:
                self.error='Native endpoint reported a data/clock discontinuity; integrity/timing acceptance refused'
    def analyze(self,*args,**kwargs):
        result=super().analyze(*args,**kwargs)
        result.update(capture_backend='native WASAPI GetBuffer/QPC; independent Python3.11 capture-only worker',
            native_packet_report=str(self.output.with_suffix('.json')),fabricated_gap_samples=0,
            recording_api='WASAPI',clock_basis='Actual WASAPI packet QPC100ns positions; no first-block or wall-clock gap estimation')
        return result
