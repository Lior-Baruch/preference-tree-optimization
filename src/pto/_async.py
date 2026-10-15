"""Sync/async bridge shared by generation, training and evaluation (no heavy imports)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code, in scripts and notebooks alike.

    With no running event loop (scripts, the CLI) this is ``asyncio.run(coro)``. Inside an
    already-running loop (e.g. Jupyter) the coroutine runs on a *separate thread* with its own
    event loop, which avoids ``nest_asyncio`` (broken on Python >= 3.13 by stricter
    ``contextvars`` re-entry rules). Every call gets a fresh event loop, so loop-bound objects
    (semaphores, locks) must be created per loop.

    The spawned thread shares the process's CUDA context; PyTorch GPU calls made from coroutines
    on that thread work because the CUDA context is per process, not per thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_future: Future = Future()

    def _thread_target() -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                result_future.set_result(result)
            except BaseException as exc:
                result_future.set_exception(exc)
            finally:
                loop.close()
        except BaseException as exc:
            if not result_future.done():
                result_future.set_exception(exc)

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join()
    return result_future.result()
