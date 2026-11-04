"""Diagnostic native WASAPI packet capture; never estimates/invents missing samples.

Uses the audited pinned SoundCard COM bindings in an isolated probe directory.
Calls GetBuffer directly to preserve actual device positions, flags and QPC.
"""
import argparse,json,sys,time,math
from pathlib import Path

def sample_budget(seconds):
    if not math.isfinite(seconds) or not 1<=seconds<=1200:raise ValueError('Capture duration must be 1–1200 seconds')
    return math.ceil(48000*seconds)+48000  # one bounded client staging buffer


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--device-name',default='CABLE Output (VB-Audio Virtual Cable)')
    args=p.parse_args();report=args.output.with_suffix('.json');ready=args.output.with_suffix('.ready');stop=args.output.with_suffix('.stop')
    try:budget=sample_budget(args.seconds)
    except ValueError as exc:p.error(str(exc))
    sys.path.insert(0,str(Path('.model-cache/audio-probe/python').resolve()))
    import numpy as np
    import soundcard as sc
    from soundcard import mediafoundation as mf
    import soundfile as sf
    if args.output.exists() or report.exists() or ready.exists():raise FileExistsError('Choose new capture paths')
    devices=[d for d in sc.all_microphones(include_loopback=False) if d.name==args.device_name and 'CABLE Output' in d.name]
    if len(devices)!=1:raise RuntimeError('Select exactly one native CABLE Output endpoint')
    packets=[];parts=[];started=time.perf_counter();count=0
    # Capture-only staging, not a playback prebuffer. QPC remains the packet's
    # actual acquisition time. A10ms client buffer lost packets when Windows
    # descheduled this diagnostic process for386ms during CPU regression tests.
    with devices[0].recorder(samplerate=48000,channels=2,blocksize=48000) as stream:
        client_buffer_frames=stream.buffersize
        ready.write_text('READY',encoding='ascii')
        while time.perf_counter()-started<args.seconds and not stop.exists():
            if not stream._capture_available_frames():time.sleep(.001);continue
            ptr=mf._ffi.new('BYTE**');frames=mf._ffi.new('UINT32*');flags=mf._ffi.new('DWORD*')
            position=mf._ffi.new('UINT64*');qpc=mf._ffi.new('UINT64*')
            result=stream._ppCaptureClient[0][0].lpVtbl.GetBuffer(stream._ppCaptureClient[0],ptr,frames,flags,position,qpc)
            mf._com.check_error(result);n=int(frames[0]);f=int(flags[0])
            try:
                if f & 2:audio=np.zeros((n,2),dtype='float32') # Actual endpoint SILENT flag, not elapsed-time inference.
                elif ptr[0]!=mf._ffi.NULL:audio=np.frombuffer(mf._ffi.buffer(ptr[0],n*8),dtype='float32').copy().reshape(n,2)
                else:raise RuntimeError('Non-silent native packet has no PCM payload')
                packets.append(dict(sample=count,frames=n,flags=f,device_position=int(position[0]),qpc_seconds=int(qpc[0])/1e7,received_at=time.perf_counter()))
                parts.append(audio);count+=n
                if count>budget:raise RuntimeError('Capture sample budget exceeded')
            finally:stream._capture_release(n)
    audio=np.concatenate(parts) if parts else np.zeros((0,2),dtype='float32')
    sf.write(args.output,audio,48000,subtype='PCM_16')
    discontinuities=[r for r in packets[1:] if r['flags']&1]
    gaps=[dict(after=a['sample'],expected=a['device_position']+a['frames'],actual=b['device_position'])
        for a,b in zip(packets,packets[1:]) if a['device_position']+a['frames']!=b['device_position']]
    report.write_text(json.dumps(dict(backend='native Windows WASAPI GetBuffer, pinned isolated SoundCard0.4.5 COM bindings',
        sample_rate=48000,channels=2,samples=count,packets=packets,discontinuities=discontinuities,position_gaps=gaps,
        fabricated_gap_samples=0,timestamp_error_packets=sum(bool(r['flags']&4) for r in packets),
        source=devices[0].name,output=str(args.output.resolve())),indent=2),encoding='utf8')
    print('NATIVE CAPTURE',count,'frames',len(discontinuities),'discontinuities',len(gaps),'position gaps',flush=True)


if __name__=='__main__':main()
