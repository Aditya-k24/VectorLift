"""
VectorLift — Timing Utilities
================================
Provides a context manager and decorators for measuring elapsed time in
milliseconds.  Designed to be ergonomic for both synchronous and async code.

Usage — context manager
------------------------
    from core.utils.timing import Timer

    with Timer() as t:
        result = do_work()

    print(f"Elapsed: {t.elapsed_ms:.2f} ms")

Usage — sync decorator
-----------------------
    from core.utils.timing import sync_timer

    @sync_timer(label="my_function")
    def my_function(x: int) -> int:
        return x * 2

    # Log output:
    # {"timestamp": "...", "level": "DEBUG", "name": "core.utils.timing",
    #  "message": "my_function completed", "elapsed_ms": 0.12, ...}

Usage — async decorator
------------------------
    from core.utils.timing import async_timer

    @async_timer(label="fetch_embeddings")
    async def fetch_embeddings(texts: list[str]) -> list[list[float]]:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from typing import Any, ParamSpec, TypeVar

from core.logging.logger import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Timer — context manager
# ---------------------------------------------------------------------------


class Timer:
    """
    Synchronous context manager that records elapsed time in milliseconds.

    Attributes
    ----------
    elapsed_ms : float
        Elapsed time in milliseconds.  Available after ``__exit__``.
        Accessing it before the block completes raises ``RuntimeError``.
    start_ns : int
        ``time.perf_counter_ns()`` value recorded at block entry.
    stop_ns : int | None
        ``time.perf_counter_ns()`` value recorded at block exit.
        ``None`` while the block is still running.

    Example
    -------
        with Timer() as t:
            expensive_operation()
        print(t.elapsed_ms)
    """

    def __init__(self, *, label: str = "timer") -> None:
        self.label = label
        self.start_ns: int = 0
        self.stop_ns: int | None = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "Timer":
        self.start_ns = time.perf_counter_ns()
        self.stop_ns = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop_ns = time.perf_counter_ns()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self.stop_ns is None:
            if self.start_ns == 0:
                raise RuntimeError("Timer has not been started yet.")
            # Called while still inside the `with` block — return live reading
            return (time.perf_counter_ns() - self.start_ns) / 1_000_000
        return (self.stop_ns - self.start_ns) / 1_000_000

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ms / 1_000

    def __repr__(self) -> str:
        if self.stop_ns is None:
            return f"Timer(label={self.label!r}, running)"
        return f"Timer(label={self.label!r}, elapsed_ms={self.elapsed_ms:.3f})"


# ---------------------------------------------------------------------------
# AsyncTimer — async context manager
# ---------------------------------------------------------------------------


class AsyncTimer:
    """
    Asynchronous context manager that records elapsed time in milliseconds.

    Example
    -------
        async with AsyncTimer() as t:
            await async_operation()
        print(t.elapsed_ms)
    """

    def __init__(self, *, label: str = "async_timer") -> None:
        self.label = label
        self.start_ns: int = 0
        self.stop_ns: int | None = None

    async def __aenter__(self) -> "AsyncTimer":
        self.start_ns = time.perf_counter_ns()
        self.stop_ns = None
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop_ns = time.perf_counter_ns()

    @property
    def elapsed_ms(self) -> float:
        if self.stop_ns is None:
            if self.start_ns == 0:
                raise RuntimeError("AsyncTimer has not been started yet.")
            return (time.perf_counter_ns() - self.start_ns) / 1_000_000
        return (self.stop_ns - self.start_ns) / 1_000_000

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ms / 1_000

    def __repr__(self) -> str:
        if self.stop_ns is None:
            return f"AsyncTimer(label={self.label!r}, running)"
        return f"AsyncTimer(label={self.label!r}, elapsed_ms={self.elapsed_ms:.3f})"


# ---------------------------------------------------------------------------
# sync_timer decorator
# ---------------------------------------------------------------------------


def sync_timer(
    func: Callable[P, R] | None = None,
    *,
    label: str | None = None,
    log: bool = True,
    log_level: str = "DEBUG",
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that times a synchronous function and optionally logs the result.

    Can be used with or without arguments::

        @sync_timer
        def my_fn(): ...

        @sync_timer(label="custom_label", log_level="INFO")
        def my_fn(): ...

    Parameters
    ----------
    func:
        The function to wrap (when the decorator is used without arguments).
    label:
        Override the log label.  Defaults to ``func.__qualname__``.
    log:
        Emit a structured log record with the elapsed time.
    log_level:
        Log level for the timing record (default: ``"DEBUG"``).

    Returns
    -------
    The wrapped function.  The wrapper has an ``elapsed_ms`` attribute set to
    the most recent elapsed time after each call.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        effective_label = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with Timer(label=effective_label) as t:
                result = fn(*args, **kwargs)
            wrapper.elapsed_ms = t.elapsed_ms  # type: ignore[attr-defined]
            if log:
                log_fn = getattr(logger, log_level.lower(), logger.debug)
                log_fn(
                    f"{effective_label} completed",
                    extra={"elapsed_ms": t.elapsed_ms, "label": effective_label},
                )
            return result

        wrapper.elapsed_ms = 0.0  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        # Called as @sync_timer without parentheses
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# async_timer decorator
# ---------------------------------------------------------------------------


def async_timer(
    func: Callable[P, Any] | None = None,
    *,
    label: str | None = None,
    log: bool = True,
    log_level: str = "DEBUG",
) -> Any:
    """
    Decorator that times an asynchronous (``async def``) function.

    Usage is identical to :func:`sync_timer`.  The wrapped coroutine can be
    awaited normally::

        @async_timer(label="embed_batch", log_level="INFO")
        async def embed(texts: list[str]) -> list[list[float]]:
            ...

        embeddings = await embed(texts)
        print(embed.elapsed_ms)
    """

    def decorator(fn: Callable[P, Any]) -> Callable[P, Any]:
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(
                f"@async_timer requires an async function; got {fn!r}"
            )
        effective_label = label or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            async with AsyncTimer(label=effective_label) as t:
                result = await fn(*args, **kwargs)
            wrapper.elapsed_ms = t.elapsed_ms  # type: ignore[attr-defined]
            if log:
                log_fn = getattr(logger, log_level.lower(), logger.debug)
                log_fn(
                    f"{effective_label} completed",
                    extra={"elapsed_ms": t.elapsed_ms, "label": effective_label},
                )
            return result

        wrapper.elapsed_ms = 0.0  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Convenience context-manager factory functions
# ---------------------------------------------------------------------------


@contextmanager
def timed_block(label: str = "block", *, log: bool = False) -> Iterator[Timer]:
    """
    Context manager factory that yields a :class:`Timer`.

    Example
    -------
        with timed_block("load_model", log=True) as t:
            model = load()
        # t.elapsed_ms is now available
    """
    with Timer(label=label) as t:
        yield t
    if log:
        logger.debug(
            f"{label} completed",
            extra={"elapsed_ms": t.elapsed_ms, "label": label},
        )


@asynccontextmanager
async def async_timed_block(
    label: str = "async_block",
    *,
    log: bool = False,
) -> AsyncIterator[AsyncTimer]:
    """
    Async context manager factory that yields an :class:`AsyncTimer`.

    Example
    -------
        async with async_timed_block("fetch_results", log=True) as t:
            results = await fetch()
    """
    async with AsyncTimer(label=label) as t:
        yield t
    if log:
        logger.debug(
            f"{label} completed",
            extra={"elapsed_ms": t.elapsed_ms, "label": label},
        )
