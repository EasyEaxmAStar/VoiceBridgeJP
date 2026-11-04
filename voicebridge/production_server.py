"""Only the installed production backends, with localhost-only network access."""
import argparse
import ipaddress
import os
import socket
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--instance",required=True)
    parser.add_argument("--gate",type=Path,required=True)
    args = parser.parse_args()
    # Parent places us in its Windows Job before permitting any model/worker
    # startup. An assignment failure cannot leave an uncontained GPU child.
    deadline = time.monotonic()+10
    while not args.gate.exists():
        if time.monotonic()>deadline:
            raise TimeoutError("Parent did not establish the process ownership gate")
        time.sleep(.01)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
    original_connect,original_ex = socket.socket.connect,socket.socket.connect_ex
    def allowed(address):
        if isinstance(address,tuple):
            host = address[0]
            return host=="localhost" or ipaddress.ip_address(host).is_loopback
        return False
    def connect(stream,address):
        if not allowed(address):
            raise RuntimeError("Production inference permits localhost sockets only")
        return original_connect(stream,address)
    def connect_ex(stream,address):
        if not allowed(address):
            raise RuntimeError("Production inference permits localhost sockets only")
        return original_ex(stream,address)
    socket.socket.connect,socket.socket.connect_ex = connect,connect_ex
    from ort.config import Config
    from ort.wav_pipeline import ASR_MODEL, MT_MODEL, load_production, require_production
    from ort.server import create_app
    import uvicorn
    cfg = Config(runtime="cuda",device="cuda:0",asr_backend="qwen3",qwen_asr_device="cuda:0",
        qwen_asr_model=ASR_MODEL,qwen_asr_offline=True,torch_dtype="bfloat16",torch_attn="sdpa",
        translator_backend="opusmt",opusmt_model=MT_MODEL,tts_backend="gptsovits",output_speed=1)
    require_production(cfg)
    app = create_app(cfg,warmup=True,pipeline_factory=load_production)
    app.state.voicebridge_instance = args.instance
    uvicorn.run(app,host="127.0.0.1",port=args.port,log_level="info",ws="websockets",
                timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
