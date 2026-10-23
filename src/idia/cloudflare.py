"""The ``cloudflare`` extra's provider: Clef and Clairic behind one router.

:func:`cloudflare` builds the Clef leaf for the label kinds and the Clairic leaf for
extraction over one HTTP client, wraps each in a :class:`~idia.dispatch.Dispatcher`
sharing the limiters given, and returns a :class:`~idia.dispatch.Router` keyed
``noul``, ``choice`` and ``score`` to Clef and ``extraction`` to Clairic. It lives outside
``providers/`` because it builds the call loop; nothing imports it but its caller.
"""

from __future__ import annotations

from typing import Final

import httpx2

from idia.dispatch import Dispatcher, Router
from idia.limits import ConcurrencyLimit, RateLimiter
from idia.providers.cloudflare import (
    Clairic,
    Clef,
    ClefModel,
    _share_client,  # pyright: ignore[reportPrivateUsage]
)
from idia.retry import RetryPolicy

__all__ = ["cloudflare"]

_DEFAULT_RETRY: Final = RetryPolicy()


def cloudflare(
    *,
    account_id: str | None = None,
    api_token: str | None = None,
    clef_model: ClefModel = "clef",
    client: httpx2.AsyncClient | None = None,
    retry: RetryPolicy = _DEFAULT_RETRY,
    concurrency: ConcurrencyLimit | None = None,
    rate: RateLimiter | None = None,
) -> Router:
    """The router over Clef and Clairic, sharing one client and the limiters given.

    ``account_id`` and ``api_token`` default to ``CLOUDFLARE_ACCOUNT_ID`` and
    ``CLOUDFLARE_API_TOKEN``. ``clef_model`` is ``clef``, ``clef-flash``,
    ``@cf/cloudflare/clef`` or ``@cf/cloudflare/clef-flash``, the Clef member reporting
    the short name either way. With no ``client``, the Clef leaf creates one, which
    Clairic shares and the router's ``aclose`` closes; it follows the running event loop,
    Clef making a fresh one on another loop and Clairic sending with whichever Clef holds.
    A ``client`` the caller injects is never closed, and is used on the loop it was first
    used on. ``retry``, ``concurrency`` and ``rate`` go to both dispatchers, the same
    limiter instances to each, so the two models draw on one allowance.
    """
    clef = Clef(model=clef_model, account_id=account_id, api_token=api_token, client=client)
    clairic = Clairic(account_id=account_id, api_token=api_token, client=clef.client)
    _share_client(clairic, clef)
    return _router(clef, clairic, retry=retry, concurrency=concurrency, rate=rate)


def _router(
    clef: Clef,
    clairic: Clairic,
    *,
    retry: RetryPolicy,
    concurrency: ConcurrencyLimit | None,
    rate: RateLimiter | None,
) -> Router:
    """The router's shape over two built leaves: one dispatcher each, the limiters shared."""
    label = Dispatcher(clef, retry=retry, concurrency=concurrency, rate=rate)
    extraction = Dispatcher(clairic, retry=retry, concurrency=concurrency, rate=rate)
    return Router({"noul": label, "choice": label, "score": label, "extraction": extraction})
