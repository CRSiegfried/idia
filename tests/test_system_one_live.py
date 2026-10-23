"""The one live run per System One model, which records the cassettes the other tests
replay. Skipped unless ``IDIA_LIVE=1``; the key comes from ``TYPESAFE_API_KEY``.

Each run records, for its configuration, a success per claimed kind, the provider half's
single-question requests and the failure (eleven score levels, one past the wire's cap),
and then what the record and the estimator are set from: for ``jev-1.13.0`` a request one
over the token limit and an image sent with the images header; for ``jev-1.14.0`` the
image request's questions asked again without the image, so the image's token cost is the
difference. Two more are recorded once each, skipped when their cassettes exist: the
minimum-side probe on ``jev-1.14.0``, sent outside the leaf, and a box question asked
through ``idia.provider``. A credentials failure stops the run at its first call.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Final

import httpx2
import pytest

import idia
from idia import (
    Deadline,
    ProviderError,
    Request,
    RequestFailed,
    Response,
    State,
    TextPart,
    WireRequest,
    WireResponse,
)
from idia.providers.system_one import (
    DEFAULT_BASE_URL,
    SystemOne,
    _body,  # pyright: ignore[reportPrivateUsage]
)
from tests._http import RecordingTransport, record_cassette, write_cassette
from tests._system_one_cases import (
    BOX_PROOF,
    BOX_PROOF_REQUEST,
    ELEVEN_LEVELS,
    FAILURE,
    IMAGE_SIDE_PROBES,
    TEXT_ONLY,
    WITH_IMAGES,
    WORDING,
    Configuration,
    check_red_square,
    side_probe,
    side_probe_path,
    wording_request,
)

pytestmark = pytest.mark.live

_SECONDS: Final = 60.0


def _key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        pytest.fail("TYPESAFE_API_KEY is not set")
    return key


def _leaf(configuration: Configuration, client: httpx2.AsyncClient | None) -> SystemOne:
    return SystemOne(api_key=_key(), client=client, **configuration.options())


async def _record(configuration: Configuration, name: str, wire_request: WireRequest) -> None:
    """One live call through the leaf, written as the configuration's cassette ``name``
    once :func:`_check` passes it."""
    await record_cassette(
        configuration.directory / f"{name}.json",
        lambda client: _leaf(configuration, client),
        wire_request,
        Deadline.after(_SECONDS),
        check=_check,
    )


def _check(side: Path, result: WireResponse | ProviderError) -> None:
    """Stop the run on a credentials failure, before the cassette reaches its path."""
    if isinstance(result, ProviderError) and result.status_code in (401, 403):
        pytest.fail(f"credentials refused: HTTP {result.status_code}; stopping")


async def _probe(configuration: Configuration, path: Path, wire_request: WireRequest) -> int:
    """One live call outside the leaf: the body the leaf would build, sent with the images
    header, so that what the leaf would refuse or never send still reaches the wire. Its
    cassette is written at ``path`` unless the credentials were refused; returns the status."""
    recorder = RecordingTransport()
    # The record's types, so an image Fit would refuse still goes with its own format.
    body, _ = _body(configuration.model, wire_request, frozenset({"image/png", "image/jpeg"}))
    headers = {
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
        "TypeSafe-Danger-Zone-Images": "true",
    }
    async with httpx2.AsyncClient(transport=recorder, timeout=_SECONDS) as client:
        await client.post(
            DEFAULT_BASE_URL + "/v1/systemone",
            content=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(),
            headers=headers,
        )
    [(request, response, content)] = recorder.exchanges
    if response.status_code in (401, 403):
        pytest.fail(f"credentials refused: HTTP {response.status_code}; stopping")
    named = _leaf(configuration, None)
    write_cassette(
        path,
        leaf=named,
        request=request,
        response=response,
        content=content,
        wire_request=wire_request,
        expected=None,
    )
    await named.aclose()
    return response.status_code


async def _run(configuration: Configuration) -> None:
    for kind in configuration.kinds:
        await _record(configuration, kind, configuration.wire_request(kind))
    for name, wire_request in configuration.singles().items():
        await _record(configuration, name, wire_request)
    state = configuration.wire_request("noul").state
    eleven = WireRequest(state=state, questions=ELEVEN_LEVELS.questions)
    await _record(configuration, FAILURE, eleven)


def test_live_jev_1_13_0() -> None:
    asyncio.run(_run(TEXT_ONLY))
    long_text = "The ground was under water and the match was called off. " * 7_500
    over = WireRequest(
        state=State(parts=(TextPart(text=long_text),)),
        questions=TEXT_ONLY.wire_request("noul").questions,
    )
    asyncio.run(_record(TEXT_ONLY, "over-limit", over))
    probe = Configuration(TEXT_ONLY.model, frozenset({"images"}))
    asyncio.run(
        _probe(probe, probe.directory / "probe-noul.json", WITH_IMAGES.wire_request("noul"))
    )


def test_live_jev_1_14_0() -> None:
    asyncio.run(_run(WITH_IMAGES))
    with_image = WITH_IMAGES.wire_request("noul")
    text_only = WireRequest(
        state=State(parts=with_image.state.parts[:1]), questions=with_image.questions
    )
    plain = Configuration(WITH_IMAGES.model, frozenset())
    asyncio.run(_record(plain, "probe-noul-text", text_only))


@pytest.mark.parametrize("configuration", [TEXT_ONLY, WITH_IMAGES], ids=lambda c: c.model)
def test_live_wording(configuration: Configuration) -> None:
    """A described ``noul`` and a structured instruction in one call, recorded once per
    model; skipped when its cassette exists."""
    if (configuration.directory / f"{WORDING}.json").exists():
        pytest.skip("the wording cassette is recorded once; it exists")
    asyncio.run(_record(configuration, WORDING, wording_request()))


@pytest.mark.parametrize(("width", "height"), IMAGE_SIDE_PROBES)
def test_live_jev_1_14_0_image_side(width: int, height: int) -> None:
    """One probe of the minimum image side, sent outside the leaf, which would refuse the
    sizes below it. Recorded once: skipped when its cassette exists."""
    path = side_probe_path(width, height)
    if path.exists():
        pytest.skip("each image-side probe is recorded once; its cassette exists")
    status = asyncio.run(_probe(WITH_IMAGES, path, side_probe(width, height)))
    print(f"{width}x{height}: HTTP {status}")


async def _box_through_provider(path: Path) -> Response:
    """One box question asked through ``idia.provider``, over a recording client;
    its cassette is written only when the wire answered and the box passes
    :func:`check_red_square`, so a wrong box is never kept."""
    recorder = RecordingTransport()
    async with httpx2.AsyncClient(transport=recorder, timeout=None) as client:
        provider = idia.provider(
            "system-one", model="jev-1.14.0", images=True, api_key=_key(), client=client
        )
        try:
            response = await provider.answer(
                Request(
                    state=BOX_PROOF_REQUEST.state,
                    questions=BOX_PROOF_REQUEST.questions,
                    deadline=Deadline.after(_SECONDS),
                )
            )
        except RequestFailed as failed:
            error = failed.response.outcomes[0].error
            status = error.status_code if error is not None else None
            pytest.fail(f"the box request failed: HTTP {status}; nothing kept")
    check_red_square(response)
    [(request, http_response, content)] = recorder.exchanges
    named = _leaf(WITH_IMAGES, None)
    write_cassette(
        path,
        leaf=named,
        request=request,
        response=http_response,
        content=content,
        wire_request=BOX_PROOF_REQUEST,
        expected=response.answers,
    )
    await named.aclose()
    return response


def test_live_jev_1_14_0_box_through_the_provider() -> None:
    """``box`` answered through the public entry point, recorded once."""
    path = WITH_IMAGES.directory / f"{BOX_PROOF}.json"
    if path.exists():
        pytest.skip("the box proof is recorded once; its cassette exists")
    response = asyncio.run(_box_through_provider(path))
    print(response.answers, response.outcomes[0].request_id)
