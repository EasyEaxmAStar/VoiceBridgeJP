"""Project-owned CPU video worker/diagnostics CLI. Never loads audio or models."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import signal
import sys
import threading
import time
from pathlib import Path

from .camera_events import CameraCommand,CameraMode
from .camera_video import VideoConfig,CaptureLoop,LatestFrame,CommandState,VideoPipeline,VirtualOutput


def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnose',action='store_true')
    parser.add_argument('--make-config',type=Path);parser.add_argument('--camera-index',type=int)
    parser.add_argument('--config',type=Path);parser.add_argument('--controlled',action='store_true')
    parser.add_argument('--smoke-seconds',type=float,default=5.)
    parser.add_argument('--synthetic',action='store_true');parser.add_argument('--demo-glitch',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.controlled and (args.synthetic or args.demo_glitch):parser.error('Production control cannot use synthetic/demo input')
    if not 0<args.smoke_seconds<=120:parser.error('Smoke duration must be within 0–120 seconds')
    import cv2,numpy as np,pyvirtualcam
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if args.diagnose or args.make_config:
        from .camera_devices import diagnostics
        report=diagnostics();report.update(python=sys.version,opencv=cv2.__version__,numpy=np.__version__,pyvirtualcam=pyvirtualcam.__version__,opencl=False,cuda_video=False)
        if args.make_config:
            matches=[d for d in report['cameras'] if d['index']==args.camera_index and not d['virtual']]
            if len(matches)!=1:parser.error('Choose an enumerated physical --camera-index; virtual input is refused')
            config=VideoConfig(matches[0]['identity'])
            with args.make_config.open('x',encoding='utf8') as stream:json.dump(asdict(config),stream,ensure_ascii=False,indent=2)
            report['created_config']=str(args.make_config)
        emit(report);return 0
    if args.config is None:parser.error('--config is required; run --diagnose and --make-config first')
    config=VideoConfig(**json.loads(args.config.read_text(encoding='utf8')))
    slot=LatestFrame();commands=CommandState();output=None;capture=None;pipeline=None
    if args.synthetic:
        class Synthetic:
            def __init__(self,config):self.config=config;self.count=0
            def read(self):
                time.sleep(1/self.config.fps);self.count+=1
                y,x=np.indices((self.config.height,self.config.width),dtype=np.uint16)
                return True,np.stack(((x+self.count)%256,(y+self.count)%256,(x+y)%256),axis=2).astype(np.uint8)
            def close(self):pass
        capture=CaptureLoop(config,slot,factory=Synthetic)
    else:capture=CaptureLoop(config,slot)
    reader=None;stop_reader=threading.Event();started=time.perf_counter()
    try:
        output=VirtualOutput(config);pipeline=VideoPipeline(config,slot,commands,output)
        signal.signal(signal.SIGINT,lambda *_:pipeline.stop.set())
        if args.controlled:
            def read_commands():
                try:
                    while not stop_reader.is_set():
                        line=sys.stdin.buffer.readline(2049)
                        if not line:break
                        if len(line)>2048 or not line.endswith(b'\n'):raise ValueError('Oversized video control frame')
                        data=json.loads(line)
                        if data.get('shutdown'):break
                        if set(data)!={'mode','revision','clean_fence','deadline'}:raise ValueError('Invalid video command fields')
                        mode=CameraMode(data['mode']);rev=data['revision'];fence=data['clean_fence'];deadline=data['deadline']
                        if type(rev) is not int or type(fence) is not int or not 0<=fence<=rev:raise ValueError('Invalid command revision')
                        if deadline is not None and (type(deadline) not in {int,float} or not np.isfinite(deadline)):raise ValueError('Invalid deadline')
                        commands.update(CameraCommand(mode,rev,fence,deadline))
                except Exception as exc:emit(dict(type='camera_error',error=str(exc)))
                finally:pipeline.stop.set()
            reader=threading.Thread(target=read_commands,name='camera-control-reader',daemon=True);reader.start()
        else:
            commands.update(CameraCommand(CameraMode.CLEAN,1,1,None))
            if args.demo_glitch:
                def demo():
                    deadline=time.perf_counter()+10
                    while slot.get() is None and time.perf_counter()<deadline:
                        if pipeline.stop.wait(.02):return
                    end=time.perf_counter()+1.5;revision=2
                    while time.perf_counter()<end:
                        commands.update(CameraCommand(CameraMode.GLITCHING,revision,1,end));revision+=1
                        if pipeline.stop.wait(.1):return
                    commands.update(CameraCommand(CameraMode.CLEAN,revision,revision,None))
                threading.Thread(target=demo,name='camera-demo-only',daemon=True).start()
        capture.start()
        emit(dict(type='camera_ready',width=config.width,height=config.height,fps=config.fps,
                  backend=config.virtual_backend,device=config.virtual_device,synthetic=args.synthetic,cuda_video=False))
        def status_report(status):
            emit(dict(type='camera_status',**status,**capture.status(),output=output.status()))
        pipeline.run(seconds=None if args.controlled else args.smoke_seconds,on_status=status_report)
        report=dict(type='camera_stopped',**pipeline.status(),capture_frames=capture.frames,capture_error=capture.error,
                    reconnects=capture.reconnects,elapsed=time.perf_counter()-started,synthetic=args.synthetic,
                    capture_first_frame_seconds=None if capture.first_frame_at is None else capture.first_frame_at-started,
                    capture_observed_fps=None if not capture.intervals else 1/float(np.mean(tuple(capture.intervals))),capture_format=capture.format,
                    capture_thread_stopped=capture.close())
        report['output']=output.status()
        emit(report)
        if args.report:
            with args.report.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
        return 0 if report['sent'] and report['capture_frames'] else 2
    except Exception as exc:
        emit(dict(type='camera_error',error=str(exc),error_code=getattr(exc,'code','UNAVAILABLE')));return 2
    finally:
        stop_reader.set()
        if capture:capture.close()
        if output:output.close()


if __name__=='__main__':
    try:code=main()
    except Exception as exc:emit(dict(type='camera_error',error=str(exc)));code=2
    raise SystemExit(code)
