"""CLI entry point: `ort-server` or `python -m ort.app`."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .config import Config
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="open-realtime-translate server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Preload and warm up models at startup to avoid first-connection cold start",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    app = create_app(Config.from_env(), warmup=args.warmup)
    uvicorn.run(app, host=args.host, port=args.port, ws="websockets")
    # Hard-exit after server shutdown to avoid SIGBUS from native MLX destructors.
    os._exit(0)


if __name__ == "__main__":
    main()
