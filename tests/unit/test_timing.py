"""
Unit tests for core/utils/timing.py.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.utils.timing import AsyncTimer, Timer, async_timer, sync_timer


# ---------------------------------------------------------------------------
# Timer (sync context manager)
# ---------------------------------------------------------------------------


class TestTimer:
    def test_timer_measures_elapsed_ms(self):
        with Timer() as t:
            time.sleep(0.1)
        # Allow generous tolerance for CI environments
        assert t.elapsed_ms >= 90, f"Expected ≥90 ms, got {t.elapsed_ms:.2f} ms"

    def test_timer_elapsed_ms_within_reasonable_bound(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms < 5_000, "Timer elapsed_ms unreasonably large"

    def test_timer_elapsed_seconds(self):
        with Timer() as t:
            time.sleep(0.1)
        assert t.elapsed_seconds >= 0.09

    def test_timer_not_started_raises(self):
        t = Timer()
        with pytest.raises(RuntimeError, match="not been started"):
            _ = t.elapsed_ms

    def test_timer_live_reading_inside_block(self):
        with Timer() as t:
            time.sleep(0.02)
            # Read elapsed while still inside – should not raise
            interim = t.elapsed_ms
        assert interim >= 0

    def test_timer_repr_running(self):
        t = Timer(label="my_timer")
        t.__enter__()
        r = repr(t)
        assert "running" in r
        t.__exit__(None, None, None)

    def test_timer_repr_stopped(self):
        with Timer(label="my_timer") as t:
            pass
        r = repr(t)
        assert "elapsed_ms" in r
        assert "my_timer" in r

    def test_timer_does_not_suppress_exceptions(self):
        with pytest.raises(ZeroDivisionError):
            with Timer():
                raise ZeroDivisionError("oops")

    def test_multiple_timers_independent(self):
        with Timer() as t1:
            time.sleep(0.05)
        with Timer() as t2:
            time.sleep(0.1)
        assert t2.elapsed_ms > t1.elapsed_ms - 20  # t2 ≥ t1 approximately


# ---------------------------------------------------------------------------
# AsyncTimer (async context manager)
# ---------------------------------------------------------------------------


class TestAsyncTimer:
    @pytest.mark.asyncio
    async def test_async_timer_measures_elapsed(self):
        async with AsyncTimer() as t:
            await asyncio.sleep(0.05)
        assert t.elapsed_ms >= 40, f"Expected ≥40 ms, got {t.elapsed_ms:.2f}"

    @pytest.mark.asyncio
    async def test_async_timer_not_started_raises(self):
        t = AsyncTimer()
        with pytest.raises(RuntimeError):
            _ = t.elapsed_ms

    @pytest.mark.asyncio
    async def test_async_timer_repr_stopped(self):
        async with AsyncTimer(label="async_test") as t:
            await asyncio.sleep(0.01)
        assert "elapsed_ms" in repr(t)


# ---------------------------------------------------------------------------
# sync_timer decorator
# ---------------------------------------------------------------------------


class TestSyncTimerDecorator:
    def test_decorator_preserves_return_value(self):
        @sync_timer
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_decorator_sets_elapsed_ms_attribute(self):
        @sync_timer(log=False)
        def noop() -> None:
            time.sleep(0.02)

        noop()
        assert hasattr(noop, "elapsed_ms")
        assert noop.elapsed_ms >= 10

    def test_decorator_without_parens(self):
        @sync_timer
        def multiply(x: int) -> int:
            return x * 2

        result = multiply(4)
        assert result == 8

    def test_decorator_with_custom_label(self):
        @sync_timer(label="my_custom_label", log=False)
        def fn() -> str:
            return "hello"

        assert fn() == "hello"

    def test_decorator_measures_correct_time(self):
        @sync_timer(log=False)
        def slow_fn() -> None:
            time.sleep(0.1)

        slow_fn()
        assert slow_fn.elapsed_ms >= 80

    def test_decorator_propagates_exception(self):
        @sync_timer(log=False)
        def failing() -> None:
            raise RuntimeError("deliberate")

        with pytest.raises(RuntimeError, match="deliberate"):
            failing()


# ---------------------------------------------------------------------------
# async_timer decorator
# ---------------------------------------------------------------------------


class TestAsyncTimerDecorator:
    @pytest.mark.asyncio
    async def test_async_decorator_preserves_return_value(self):
        @async_timer(log=False)
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await greet("World")
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_async_decorator_sets_elapsed_ms_attribute(self):
        @async_timer(log=False)
        async def sleep_fn() -> None:
            await asyncio.sleep(0.05)

        await sleep_fn()
        assert hasattr(sleep_fn, "elapsed_ms")
        assert sleep_fn.elapsed_ms >= 30

    @pytest.mark.asyncio
    async def test_async_decorator_without_parens(self):
        @async_timer
        async def identity(x: int) -> int:
            return x

        assert await identity(42) == 42

    @pytest.mark.asyncio
    async def test_async_decorator_propagates_exception(self):
        @async_timer(log=False)
        async def failing() -> None:
            raise ValueError("async failure")

        with pytest.raises(ValueError, match="async failure"):
            await failing()

    def test_async_timer_on_sync_function_raises(self):
        with pytest.raises(TypeError, match="async function"):

            @async_timer(log=False)
            def not_async() -> None:
                pass
