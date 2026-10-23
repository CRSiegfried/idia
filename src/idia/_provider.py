"""``idia.provider``: a registered leaf, built by name and wrapped in a dispatcher.

Each registered name has its own overload, typed with its leaf's constructor keywords, so a
type checker refuses an option the leaf does not take. The overloads exist for the checker
alone, under ``TYPE_CHECKING``, and name no SDK type: a ``client`` is typed ``object``, so a
program without an extra type-checks as one with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from idia.dispatch import Dispatcher
from idia.limits import ConcurrencyLimit, RateLimiter
from idia.providers import _load_leaf  # pyright: ignore[reportPrivateUsage]
from idia.retry import RetryPolicy

if TYPE_CHECKING:
    from typing_extensions import deprecated

    from idia.core import Capabilities
    from idia.providers.cloudflare import ClefModel
    from idia.providers.system_one import SystemOneModel

_DEFAULT_RETRY = RetryPolicy()

if TYPE_CHECKING:

    @overload
    def provider(
        name: Literal["system-one"],
        *,
        model: SystemOneModel = ...,
        api_key: str | None = ...,
        base_url: str = ...,
        client: object | None = ...,
        long_context: bool = ...,
        images: bool = ...,
        capabilities: Capabilities | None = ...,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
    ) -> Dispatcher: ...

    @overload
    def provider(
        name: Literal["openai-decisions"],
        *,
        model: str = ...,
        api_key: str | None = ...,
        client: object | None = ...,
        max_questions: int = ...,
        capabilities: Capabilities | None = ...,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
    ) -> Dispatcher: ...

    @overload
    def provider(
        name: Literal["openai-chat"],
        *,
        model: str,
        api_key: str | None = ...,
        client: object | None = ...,
        elicitation: str = ...,
        max_off_label_mass: float | None = ...,
        capabilities: Capabilities | None = ...,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
    ) -> Dispatcher: ...

    @overload
    def provider(
        name: Literal["cloudflare-clef"],
        *,
        model: ClefModel = ...,
        account_id: str | None = ...,
        api_token: str | None = ...,
        client: object | None = ...,
        capabilities: Capabilities | None = ...,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
    ) -> Dispatcher: ...

    @overload
    def provider(
        name: Literal["cloudflare-clairic"],
        *,
        model: Literal["clairic", "@cf/cloudflare/clairic"] = ...,
        account_id: str | None = ...,
        api_token: str | None = ...,
        client: object | None = ...,
        capabilities: Capabilities | None = ...,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
    ) -> Dispatcher: ...

    @overload
    @deprecated(
        "a registered provider takes its leaf's keywords, each of its type and every required "
        "one given, and retry, concurrency and rate; any other call raises TypeError"
    )
    def provider(
        name: Literal[
            "system-one",
            "openai-decisions",
            "openai-chat",
            "cloudflare-clef",
            "cloudflare-clairic",
        ],
        *,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
        **options: object,
    ) -> Dispatcher: ...

    @overload
    def provider(
        name: str,
        *,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
        **options: object,
    ) -> Dispatcher: ...


def provider(
    name: str,
    *,
    retry: RetryPolicy = _DEFAULT_RETRY,
    concurrency: ConcurrencyLimit | None = None,
    rate: RateLimiter | None = None,
    **options: object,
) -> Dispatcher:
    """The provider registered as ``name``: a :class:`~idia.dispatch.Dispatcher` over
    its leaf, built from ``options``, with the given policy and limits.

    The re-ask after a fault stays off; build a ``Dispatcher`` by hand to turn it on. A
    keyword the leaf does not take raises ``TypeError`` naming this call and the registered
    name.
    """
    leaf = _load_leaf(name, **options)
    return Dispatcher(leaf, retry=retry, concurrency=concurrency, rate=rate)
