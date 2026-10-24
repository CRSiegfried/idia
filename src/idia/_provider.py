"""``idia.provider``: a registered leaf, built by name and wrapped in a dispatcher;
``idia.leaf``: the same leaf, unwrapped; ``idia.provider_names``: the registered names.

Each registered name has its own overload, typed with its leaf's constructor keywords, so a
type checker refuses an option the leaf does not take. The overloads exist for the checker
alone, under ``TYPE_CHECKING``, and name no SDK type: a ``client`` is typed ``object``, so a
program without an extra type-checks as one with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from idia.core import Leaf
from idia.dispatch import Dispatcher
from idia.limits import ConcurrencyLimit, RateLimiter
from idia.providers import (  # pyright: ignore[reportPrivateUsage]
    _load_leaf,  # pyright: ignore[reportPrivateUsage]
    _registered,  # pyright: ignore[reportPrivateUsage]
)
from idia.retry import RetryPolicy

if TYPE_CHECKING:
    from typing_extensions import deprecated

    from idia.core import Capabilities
    from idia.providers._gliner import GlinerRuntime
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
    def provider(
        name: Literal["gliner"],
        *,
        model: Literal["fastino/gliner2.5-small-v1"] = ...,
        revision: str | None = ...,
        threshold: float = ...,
        runtime: GlinerRuntime | None = ...,
        retry: RetryPolicy = ...,
        concurrency: ConcurrencyLimit | None = ...,
        rate: RateLimiter | None = ...,
    ) -> Dispatcher: ...

    @overload
    @deprecated(
        "a registered provider takes its leaf's keywords, each of its type and every required "
        "one given, and retry, concurrency and rate; an unknown or missing keyword raises TypeError"
    )
    def provider(
        name: Literal[
            "system-one",
            "openai-decisions",
            "openai-chat",
            "cloudflare-clef",
            "cloudflare-clairic",
            "gliner",
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

    @overload
    def leaf(
        name: Literal["system-one"],
        *,
        model: SystemOneModel = ...,
        api_key: str | None = ...,
        base_url: str = ...,
        client: object | None = ...,
        long_context: bool = ...,
        images: bool = ...,
        capabilities: Capabilities | None = ...,
    ) -> Leaf: ...

    @overload
    def leaf(
        name: Literal["openai-decisions"],
        *,
        model: str = ...,
        api_key: str | None = ...,
        client: object | None = ...,
        max_questions: int = ...,
        capabilities: Capabilities | None = ...,
    ) -> Leaf: ...

    @overload
    def leaf(
        name: Literal["openai-chat"],
        *,
        model: str,
        api_key: str | None = ...,
        client: object | None = ...,
        elicitation: str = ...,
        max_off_label_mass: float | None = ...,
        capabilities: Capabilities | None = ...,
    ) -> Leaf: ...

    @overload
    def leaf(
        name: Literal["cloudflare-clef"],
        *,
        model: ClefModel = ...,
        account_id: str | None = ...,
        api_token: str | None = ...,
        client: object | None = ...,
        capabilities: Capabilities | None = ...,
    ) -> Leaf: ...

    @overload
    def leaf(
        name: Literal["cloudflare-clairic"],
        *,
        model: Literal["clairic", "@cf/cloudflare/clairic"] = ...,
        account_id: str | None = ...,
        api_token: str | None = ...,
        client: object | None = ...,
        capabilities: Capabilities | None = ...,
    ) -> Leaf: ...

    @overload
    def leaf(
        name: Literal["gliner"],
        *,
        model: Literal["fastino/gliner2.5-small-v1"] = ...,
        revision: str | None = ...,
        threshold: float = ...,
        runtime: GlinerRuntime | None = ...,
    ) -> Leaf: ...

    @overload
    @deprecated(
        "a registered leaf takes its own keywords, each of its type and every required one "
        "given; an unknown or missing keyword raises TypeError"
    )
    def leaf(
        name: Literal[
            "system-one",
            "openai-decisions",
            "openai-chat",
            "cloudflare-clef",
            "cloudflare-clairic",
            "gliner",
        ],
        **options: object,
    ) -> Leaf: ...

    @overload
    def leaf(
        name: str,
        **options: object,
    ) -> Leaf: ...


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
    built = _load_leaf(name, "provider", **options)
    return Dispatcher(built, retry=retry, concurrency=concurrency, rate=rate)


def leaf(name: str, **options: object) -> Leaf:
    """The leaf registered as ``name``, built from ``options`` and not wrapped: the leaf
    ``idia.provider(name, **options)`` wraps. Wrap it in ``Dispatcher(leaf, ...)`` to ask it
    anything, around a leaf of your own if you like, such as one counting every call.

    An unknown name raises ``ProviderNotFound``, and a missing extra ``ModuleNotFoundError``
    naming it. A keyword the leaf does not take raises ``TypeError`` naming this call and the
    name; for ``retry``, ``concurrency``, ``rate``, ``estimator``, ``overflow`` or ``reask``
    it adds that they are the ``Dispatcher``'s.
    """
    return _load_leaf(name, "leaf", **options)


def provider_names() -> tuple[str, ...]:
    """The names ``idia.provider`` and ``idia.leaf`` accept, sorted: the names
    ``ProviderNotFound`` lists. A name is listed whether or not its extra is installed."""
    return _registered()
