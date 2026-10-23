"""The limits a dispatcher waits on before each call: a concurrency limit and a rate limiter.

Both waits are bounded by the request's deadline. A limiter whose wait cannot end before
the deadline raises :class:`~idia.core.DeadlineExceeded`. One instance may be shared
by two dispatchers whose leaves share one vendor limit.
"""

from __future__ import annotations

import asyncio
import time

from idia.core import Deadline, DeadlineExceeded


class ConcurrencyLimit:
    """At most ``limit`` calls in flight at once, across every dispatcher that shares it.

    A slot is held across one call only: the dispatcher releases it before a retry sleep
    and takes it again for the next attempt. It wraps an :class:`asyncio.Semaphore` made for
    the running event loop and made again when the loop changes, so it serves successive
    ``asyncio.run`` calls; the limit holds within one loop.
    """

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit is at least 1")
        self.limit = limit
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def _for_running_loop(self) -> asyncio.Semaphore:
        """The semaphore of the running loop, made fresh when the loop has changed."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.limit)
        return self._semaphore

    async def acquire(self, deadline: Deadline) -> None:
        """Take a slot, waiting no longer than the deadline allows."""
        semaphore = self._for_running_loop()
        try:
            async with asyncio.timeout(max(deadline.remaining(), 0.0)):
                await semaphore.acquire()
        except TimeoutError:
            raise DeadlineExceeded(
                "the deadline passed while waiting for a concurrency slot"
            ) from None

    def release(self) -> None:
        """Give the slot back."""
        assert self._semaphore is not None, "release without acquire"
        self._semaphore.release()


class RateLimiter:
    """A token bucket of ``rate`` calls per ``per`` seconds; one token per attempt.

    The bucket starts full, so a burst of ``rate`` calls goes through at once. When the
    bucket is empty the caller waits for the next token, unless that wait would end after
    the deadline, in which case it raises at once, consuming nothing.
    """

    def __init__(self, rate: float, per: float = 1.0) -> None:
        if rate <= 0 or per <= 0:
            raise ValueError("rate and per are above 0")
        self.rate = float(rate)
        self.per = float(per)
        self._tokens = self.rate
        self._updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.rate, self._tokens + (now - self._updated) * self.rate / self.per)
        self._updated = now

    async def acquire(self, deadline: Deadline) -> None:
        """Take one token, waiting for it only when the wait ends before the deadline."""
        self._refill()
        wait = 0.0 if self._tokens >= 1 else (1 - self._tokens) * self.per / self.rate
        if wait > deadline.remaining():
            raise DeadlineExceeded(
                f"the rate limiter's next token is {wait:.3f}s away, past the deadline"
            )
        self._tokens -= 1
        if wait > 0:
            await asyncio.sleep(wait)
