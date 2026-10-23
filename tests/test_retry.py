"""The retry policy: attempts count calls, the delay follows the policy."""

from __future__ import annotations

import dataclasses

import pytest

from idia import RetryPolicy


def test_retry_policy_defaults_and_is_frozen() -> None:
    policy = RetryPolicy()
    assert (policy.attempts, policy.base_delay, policy.jitter, policy.honour_retry_after) == (
        3,
        0.5,
        0.25,
        True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.attempts = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempts": 0},
        {"attempts": True},
        {"base_delay": -0.1},
        {"jitter": -0.1},
        {"jitter": 1.5},
    ],
)
def test_retry_policy_refuses_bad_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"at least|fraction"):
        RetryPolicy(**overrides)  # type: ignore[arg-type]


def test_retry_after_is_used_as_given() -> None:
    honouring = RetryPolicy(base_delay=10.0, jitter=0.0)
    assert honouring.delay(1, 7.5) == 7.5
    assert honouring.delay(1, 0.0) == 0.0
    ignoring = RetryPolicy(base_delay=10.0, jitter=0.0, honour_retry_after=False)
    assert ignoring.delay(1, 7.5) == 10.0


def test_delay_doubles_per_failure_without_jitter() -> None:
    policy = RetryPolicy(base_delay=0.5, jitter=0.0)
    assert [policy.delay(n, None) for n in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 4.0]


def test_delay_jitter_stays_within_its_spread() -> None:
    policy = RetryPolicy(base_delay=1.0, jitter=0.25)
    for _ in range(200):
        assert 0.75 <= policy.delay(1, None) <= 1.25
        assert 1.5 <= policy.delay(2, None) <= 2.5


def test_delay_follows_at_least_one_failure() -> None:
    with pytest.raises(ValueError, match="at least one failure"):
        RetryPolicy().delay(0, None)


def test_without_delay_retries_with_no_wait() -> None:
    policy = RetryPolicy.without_delay()
    assert policy == RetryPolicy(attempts=3, base_delay=0.0, jitter=0.0, honour_retry_after=False)
    assert policy.delay(1, 30.0) == 0.0
    assert policy.delay(3, None) == 0.0
    assert RetryPolicy.without_delay(attempts=1).attempts == 1
