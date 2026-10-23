"""The retry policy: how many calls a wire request gets and how long to wait between them.

The dispatcher applies it to a :class:`~idia.core.ProviderError` whose ``is_retryable``
is true, and to nothing else: a fault in an answer never reaches it. The policy only says
how long to wait; the dispatcher refuses any wait that would overrun the deadline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many calls a wire request gets, and the delay before each retry.

    ``attempts`` counts calls, so ``attempts=1`` never retries. The delay before the retry
    after ``failures`` failed calls is ``retry_after`` when the error carried one and
    ``honour_retry_after`` is set, used exactly as given; otherwise
    ``base_delay * 2 ** (failures - 1)``, spread by up to ``jitter`` either way.
    """

    attempts: int = 3
    base_delay: float = 0.5
    jitter: float = 0.25
    honour_retry_after: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.attempts, bool) or self.attempts < 1:
            raise ValueError("attempts is at least 1: every wire request gets one call")
        if self.base_delay < 0:
            raise ValueError("base_delay is at least 0")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter is a fraction between 0 and 1")

    @classmethod
    def without_delay(cls, attempts: int = 3) -> RetryPolicy:
        """A policy that makes up to ``attempts`` calls with no wait between them: no base
        delay, no jitter, ``retry_after`` ignored. The form a replay and the conformance suite use,
        so a recorded retry does not sleep again."""
        return cls(attempts=attempts, base_delay=0.0, jitter=0.0, honour_retry_after=False)

    def delay(self, failures: int, retry_after: float | None) -> float:
        """Seconds to wait before the next call, after ``failures`` failed ones."""
        if failures < 1:
            raise ValueError("a delay follows at least one failure")
        if self.honour_retry_after and retry_after is not None:
            return retry_after
        spread = 1 + self.jitter * (2 * random.random() - 1)
        return self.base_delay * 2 ** (failures - 1) * spread
