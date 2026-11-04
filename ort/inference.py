"""Process-wide single dedicated inference thread.

MLX/Metal must initialize and run inference on the same fixed thread. Calling it
from different asyncio default ThreadPoolExecutor workers can hang; this was
observed when routing WS/TestClient inference through `asyncio.to_thread`.

For that reason, model construction, warmup, transcribe, translate, and synthesize
all run through this one serialized worker. The rule applies to uvicorn and tests.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Callable, TypeVar

_T = TypeVar("_T")

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ort-infer"
)


def executor() -> concurrent.futures.ThreadPoolExecutor:
    return _EXECUTOR


async def run(fn: Callable[..., _T], *args: object) -> _T:
    """Run fn on the dedicated inference thread and await the result."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, fn, *args)


def run_sync(fn: Callable[..., _T], *args: object) -> _T:
    """Run fn synchronously on the dedicated inference thread."""
    return _EXECUTOR.submit(fn, *args).result()
