"""The one live run per Cloudflare model, which records the cassettes the other tests
replay. Skipped unless ``IDIA_LIVE=1``; the credentials come from
``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN``, and the account id is written into
every cassette as the placeholder the replays build the leaves with.

Clef, per model: a success per kind, the provider half's singles, an image, a long state
asked one and three questions (whether the state is counted once, from usage), and the
failure (65 questions, one past the cap). Clairic: the provider half's request, which is
also its kind cassette, and its singles, a state with characters outside the BMP before
the target (the span unit), the two value types the model page does not show, and one raw
call outside the leaf asking for an ``estimate``, whose reply is the candidate-shape
cassette; then, as the test right after it, so a whole-file run records the same cassettes
in the same order and the past-context call still comes last, Clairic's refusals (``-k refusals``):
the field-name probe, a raw call sending a field named ``1bad``, recorded once and never
overwritten, and the past-context call, a request Fit refuses (a state past the context
for one pass) sent to the leaf directly, likewise recorded once; and, as its own test,
the extraction (text, a number with decimals and a unit in its instruction, a date without
a year under ``reference_date``, a list, a duration, and a field the state does not
determine). A test per model, selected by ``-k order``, records the question-order
probe once: one request sent twice unchanged and once with its questions reversed, into
the model's ``order/`` subdirectory; it skips when the three cassettes exist. A test per
model, selected by ``-k alias``, records a request whose keys no
wire pattern takes, which the leaf sends under ``q0..qN`` or ``f0..fN``: the evidence that the
aliases are honoured on the wire. A credentials failure stops the run; so does a model
being gone, its reply kept as ``gone/gone.json``, never overwritten, and any call that
fails when it was meant to succeed or the reverse. Each call is recorded through a check
on a side file, so a call that stops the run leaves the committed cassette as it was and
no failure is ever stored under a success's name. Every text span recorded must slice the
state to its value, checked on the side file too, so a span that does not keeps the
side file and leaves the committed cassette as it was.

A test selected by ``-k question_id`` records Clef's question-id probe once, on ``clef``:
a raw call whose one question has the id ``match:intent``, outside the wire's pattern, its
body otherwise a recorded success's.

A raw call's cassette is written only once its status is the evidence wanted: a 200 for
the estimate; a 4xx for the field-name and question-id probes, whose 200 means the wire
took the name or id and fails the run, and whose 429 or 5xx is no evidence and stops it,
nothing written; a model gone, a 404 or code 3042 or 5007, stops it too.

A real Clairic refusal of a request the leaf sends does not exist yet. The refusals' run
of 2026-10-22 recorded the field-name probe (422, code 3030), and the wire answered
:data:`~tests._cloudflare_cases.CLAIRIC_FAILURE`, its state past the context, with
``truncated: true`` rather than refusing it (U7). So the probe is Clairic's failure
cassette, and the answer is kept as evidence at
:data:`~tests._cloudflare_cases.PAST_CONTEXT_TRUNCATED`. The past-context call is recorded
to a side file outside the cassette directory and written there only when the wire answers
it truncated; a 429, a 5xx or no response is no evidence and stops the run, nothing kept;
any other outcome, an answer not truncated or a refusal, keeps the side file, stops the
run and reports its path: a result to investigate, since it contradicts the evidence.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

import httpx2
import pytest

from idia import (
    Deadline,
    ExtractionAnswer,
    ProviderError,
    Question,
    Span,
    TextPart,
    WireRequest,
    WireResponse,
)
from idia.providers.cloudflare import Clairic, Clef
from tests._cloudflare_cases import (
    ACCOUNT,
    ACCOUNT_ENV,
    ALIAS,
    CANDIDATE,
    CLAIRIC_DIR,
    CLAIRIC_FAILURE,
    CLAIRIC_PROVIDER,
    CLEF_FAILURE,
    CLEF_ORDER,
    EMOJI,
    ESTIMATE_BODY,
    EXTRACTION,
    FAILURE,
    FIELD_NAME_BODY,
    FIELD_NAME_PROBE,
    FIELD_NAME_REQUEST,
    GONE,
    IMAGE_REQUEST,
    ORDER,
    ORDER_NAMES,
    PAST_CONTEXT_TRUNCATED,
    QUESTION_ID_BODY,
    QUESTION_ID_PROBE,
    QUESTION_ID_REQUEST,
    RECOUNT_ONE,
    RECOUNT_THREE,
    TEXT,
    TIMES,
    TOKEN_ENV,
    WORDING,
    aliased,
    clef_dir,
    clef_request,
    order_requests,
    singles,
    wording_request,
)
from tests._http import (
    NoExchange,
    RecordingTransport,
    _write_atomic,  # pyright: ignore[reportPrivateUsage]
    load_cassette,
    parse_body,
    record_cassette,
    write_cassette,
)

pytestmark = pytest.mark.live

_SECONDS: Final = 60.0
_GONE_CODES: Final = ("3042", "5007")


def _credentials() -> tuple[str, str]:
    account = os.environ.get(ACCOUNT_ENV)
    token = os.environ.get(TOKEN_ENV)
    if not account or not token:
        pytest.fail(f"{ACCOUNT_ENV} and {TOKEN_ENV} must be set")
    return account, token


def _redact() -> dict[str, str]:
    return {_credentials()[0]: ACCOUNT}


async def _record(
    path: Path,
    factory: Callable[[httpx2.AsyncClient], Clef | Clairic],
    wire_request: WireRequest,
    *,
    spans: bool = False,
) -> WireResponse | ProviderError:
    """One live call through the leaf, written at ``path`` once :func:`_check` passes it,
    and with ``spans`` once every text span slices the state to its value
    (:func:`_slices`); a call that stops the run leaves ``path`` as it was."""

    def check(side: Path, result: WireResponse | ProviderError) -> None:
        _check(path, side, result)
        if spans and isinstance(result, WireResponse):
            wrong = _slices(wire_request, result)
            if wrong:
                pytest.fail(f"spans that do not slice to their values, as code points: {wrong}")

    return await record_cassette(
        path, factory, wire_request, Deadline.after(_SECONDS), redact=_redact(), check=check
    )


def _check(path: Path, side: Path, result: WireResponse | ProviderError) -> None:
    """The check on ``side``, the call's cassette before it reaches ``path``: stop the run
    on a credentials failure, a model gone, or an outcome of the wrong kind, the cassette
    at ``path`` left as it was. Only Clef records a cassette meant to fail; Clairic's
    refusals are raw or side-file calls with checks of their own."""
    if isinstance(result, ProviderError) and result.status_code in (401, 403):
        side.unlink()
        pytest.fail(f"credentials refused: HTTP {result.status_code}; stopping")
    if _gone(result):
        kept = _keep_gone(path, side)
        pytest.fail(f"the model is gone: {result}; kept at {kept}; stopping")
    meant_to_fail = path.stem == FAILURE
    if isinstance(result, ProviderError) != meant_to_fail:
        wanted = "a failure" if meant_to_fail else "a success"
        pytest.fail(f"{path.name} was meant to record {wanted} and got {result!r}; stopping")


def _keep_gone(path: Path, side: Path) -> Path:
    """Write ``side``, a model-gone reply recorded for the case at ``path``, to
    ``gone/gone.json`` beside ``path``, which no replay reads, through the same atomic
    write as a cassette; then remove ``side`` and its directory, and return where it went.
    Refuses, stopping the run with ``side`` kept, when a ``gone.json`` is already there:
    an earlier one is never overwritten."""
    gone = path.parent / GONE / f"{GONE}.json"
    if gone.exists():
        pytest.fail(f"{gone} already holds a model gone; not overwritten; stopping")
    _write_atomic(gone, side.read_text(encoding="utf-8"))
    side.unlink()
    with contextlib.suppress(OSError):
        side.parent.rmdir()
    return gone


def _gone(result: WireResponse | ProviderError) -> bool:
    """Clairic or a Clef model withdrawn: a 404, or code 3042 or 5007."""
    return isinstance(result, ProviderError) and (
        result.status_code == 404 or any(code in result.message for code in _GONE_CODES)
    )


def _clef_factory(model: str) -> Callable[[httpx2.AsyncClient], Clef]:
    def factory(client: httpx2.AsyncClient) -> Clef:
        return Clef(model=model, client=client)  # type: ignore[arg-type]

    return factory


async def _clef_run(model: str) -> None:
    directory = clef_dir(model)
    factory = _clef_factory(model)
    for kind in ("noul", "choice", "score"):
        await _record(directory / f"{kind}.json", factory, clef_request(kind))
    for name, wire_request in singles(clef_request("choice")).items():
        await _record(directory / f"{name}.json", factory, wire_request)
    await _record(directory / "image.json", factory, IMAGE_REQUEST)
    await _record(directory / "recount-one.json", factory, RECOUNT_ONE)
    await _record(directory / "recount-three.json", factory, RECOUNT_THREE)
    await _record(directory / f"{FAILURE}.json", factory, CLEF_FAILURE)


def test_live_clef() -> None:
    asyncio.run(_clef_run("clef"))


def test_live_clef_flash() -> None:
    asyncio.run(_clef_run("clef-flash"))


@pytest.mark.parametrize("model", ["clef", "clef-flash"])
def test_live_clef_alias(model: str) -> None:
    path = clef_dir(model) / f"{ALIAS}.json"
    result = asyncio.run(_record(path, _clef_factory(model), aliased(clef_request("choice"))))
    assert isinstance(result, WireResponse), result


@pytest.mark.parametrize("model", ["clef", "clef-flash"])
def test_live_clef_wording(model: str) -> None:
    """A described ``noul`` and a structured instruction in one call, recorded once per
    model; skipped when its cassette exists."""
    path = clef_dir(model) / f"{WORDING}.json"
    if path.exists():
        pytest.skip("the wording cassette is recorded once; it exists")
    result = asyncio.run(_record(path, _clef_factory(model), wording_request()))
    assert isinstance(result, WireResponse), result


async def _order(
    directory: Path,
    factory: Callable[[httpx2.AsyncClient], Clef | Clairic],
    wire_request: WireRequest,
) -> None:
    """The question-order probe, recorded once into ``directory/order/``: ``wire_request``
    twice, then reversed, each checked as every recorded call is. Skipped when its three
    cassettes exist; a call that stops the run removes every one this run wrote, so the
    three are only ever kept together."""
    paths = [directory / ORDER / f"{name}.json" for name in ORDER_NAMES]
    if all(path.exists() for path in paths):
        pytest.skip("the question-order probe is recorded once; its three cassettes exist")
    sent: list[Path] = []
    try:
        for path, request in zip(paths, order_requests(wire_request).values(), strict=True):
            result = await _record(path, factory, request)
            sent.append(path)
            assert isinstance(result, WireResponse), result
            print(f"{path.relative_to(directory)}: {result.answers}")
    except BaseException:
        for path in sent:
            path.unlink(missing_ok=True)
        raise


@pytest.mark.parametrize("model", ["clef", "clef-flash"])
def test_live_clef_order(model: str) -> None:
    asyncio.run(_order(clef_dir(model), _clef_factory(model), CLEF_ORDER))


def _clairic(client: httpx2.AsyncClient) -> Clairic:
    return Clairic(client=client)


async def _raw(
    name: str,
    body: object,
    wire_request: WireRequest,
    *,
    refusal: bool,
    estimate: str | None = None,
    clef: bool = False,
) -> int:
    """One raw call sent outside the leaf, to Clairic or, when ``clef``, to ``clef``,
    written as the cassette ``name`` in that model's directory with no answers only when it
    is the evidence wanted: a 4xx when ``refusal``, else a 200, whose answer to the field
    ``estimate``, when named, is an estimate; its status. A credentials failure, a model
    gone (a 404, or code 3042 or 5007 on any status), a 429 or 5xx, and an outcome of the
    wrong kind each stop the run with nothing written."""
    account, token = _credentials()
    recorder = RecordingTransport()
    model = "clef" if clef else "clairic"
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
    async with httpx2.AsyncClient(transport=recorder, timeout=_SECONDS) as client:
        await client.post(
            url,
            content=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
    [(request, response, content)] = recorder.exchanges
    status = response.status_code
    if status in (401, 403):
        pytest.fail(f"credentials refused: HTTP {status}; stopping")
    shown = json.dumps(parse_body(content))[:600]
    if status == 404 or any(f'"code": {code}' in shown for code in _GONE_CODES):
        pytest.fail(
            f"{name}: the model is gone (HTTP {status}); nothing written; stopping: {shown}"
        )
    if status == 429 or status >= 500:
        pytest.fail(f"{name}: HTTP {status} is no evidence; nothing written; stopping: {shown}")
    if refusal and status == 200:
        pytest.fail(f"{name}: the wire took the request (HTTP 200); nothing written: {shown}")
    if (refusal and not 400 <= status < 500) or (not refusal and status != 200):
        pytest.fail(f"{name}: HTTP {status}, not the evidence wanted; nothing written: {shown}")
    if estimate is not None and not _holds_estimate(parse_body(content), estimate):
        pytest.fail(f"{name}: the 200 holds no estimate for {estimate!r}; nothing written: {shown}")
    print(f"{name}: HTTP {status}: {shown}")
    named = (
        Clef(model="clef", account_id=ACCOUNT, api_token="unused")
        if clef
        else Clairic(account_id=ACCOUNT, api_token="unused")
    )
    write_cassette(
        (clef_dir("clef") if clef else CLAIRIC_DIR) / f"{name}.json",
        leaf=named,
        request=request,
        response=response,
        content=content,
        wire_request=wire_request,
        expected=None,
        redact=_redact(),
    )
    await named.aclose()
    return status


def _holds_estimate(body: object, field: str) -> bool:
    """Whether a Clairic envelope answers ``field`` with an object of type ``estimate``."""
    held: object = body
    for key in ("result", "answers", field):
        if not isinstance(held, dict):
            return False
        held = cast(dict[str, object], held).get(key)
    return isinstance(held, dict) and cast(dict[str, object], held).get("type") == "estimate"


async def _estimate() -> None:
    """The raw call asking for an ``estimate``, which the leaf cannot ask for; its reply
    is the candidate-shape cassette, written only when the 200 holds an estimate. Recorded
    once, never overwritten."""
    if (CLAIRIC_DIR / f"{CANDIDATE}.json").exists():
        return
    crowd = Question(
        key="crowd",
        kind="extraction",
        instruction="The number of spectators at the ground.",
        value_type="number",
    )
    await _raw(
        CANDIDATE,
        ESTIMATE_BODY,
        WireRequest(state=TEXT, questions=(crowd,)),
        refusal=False,
        estimate="crowd",
    )


async def _field_name_probe() -> None:
    """The raw call with a field named ``1bad``, which the leaf would send as ``f0``:
    whether the wire refuses the name. Recorded once, never overwritten; a 200 means the
    leaf's pattern is wrong, and fails the run."""
    if (CLAIRIC_DIR / f"{FIELD_NAME_PROBE}.json").exists():
        return
    await _raw(FIELD_NAME_PROBE, FIELD_NAME_BODY, FIELD_NAME_REQUEST, refusal=True)


async def _past_context() -> None:
    """:data:`CLAIRIC_FAILURE` sent to the leaf directly, its cassette checked on a side
    file outside the cassette directory and written at the past-context cassette only
    once the wire has answered it truncated. Recorded once, never overwritten."""
    if PAST_CONTEXT_TRUNCATED.exists():
        return
    usage: list[object] = []

    def truncated(side: Path, result: WireResponse | ProviderError) -> None:
        if isinstance(result, ProviderError):
            status = result.status_code
            if status is None or status in (401, 403, 429) or status >= 500:
                side.unlink()
                pytest.fail(f"the past-context call got {result!r}: no evidence; nothing kept")
            pytest.fail(f"the wire refused the past-context call: {result}")
        body = load_cassette(side).response["body"]
        answered = cast(dict[str, Any], body)["result"]
        if answered.get("truncated") is not True:
            pytest.fail("the past-context call was answered whole")
        usage.append(answered.get("usage"))

    try:
        await record_cassette(
            PAST_CONTEXT_TRUNCATED,
            _clairic,
            CLAIRIC_FAILURE,
            Deadline.after(_SECONDS),
            redact=_redact(),
            check=truncated,
        )
    except NoExchange as error:
        pytest.fail(f"the past-context call got no response ({error}): no evidence")
    print(f"past-context: truncated, usage {usage[0]}")


async def _refusals() -> None:
    await _field_name_probe()
    await _past_context()


def _slices(wire_request: WireRequest, result: WireResponse) -> list[str]:
    """Every text answer whose span does not slice the state's text to its value."""
    wrong: list[str] = []
    by_key = {question.key: question for question in wire_request.questions}
    for key, answer in result.answers.items():
        if not isinstance(answer, ExtractionAnswer) or by_key[key].value_type != "text":
            continue
        values = answer.value if isinstance(answer.value, tuple) else (answer.value,)
        spans = answer.span if isinstance(answer.span, tuple) else (answer.span,)
        for value, span in zip(values, spans, strict=False):
            if value is None or not isinstance(span, Span):
                continue
            part = wire_request.state.parts[span.part]
            assert isinstance(part, TextPart)
            if part.text[span.start : span.end] != value:
                wrong.append(f"{key}: {part.text[span.start : span.end]!r} != {value!r}")
    return wrong


async def _clairic_call(
    name: str, wire_request: WireRequest, *, spans: bool = False
) -> WireResponse | ProviderError:
    """One recorded Clairic call, checked as every recorded call is, and with ``spans``
    its text spans too, before the cassette reaches its path."""
    return await _record(CLAIRIC_DIR / f"{name}.json", _clairic, wire_request, spans=spans)


async def _clairic_run() -> None:
    await _clairic_call("provider", CLAIRIC_PROVIDER, spans=True)
    for name, wire_request in singles(CLAIRIC_PROVIDER).items():
        await _clairic_call(name, wire_request)
    await _clairic_call("emoji", EMOJI, spans=True)
    await _clairic_call("times", TIMES)
    await _estimate()


async def _question_id_probe() -> None:
    """The raw call to ``clef`` with one question under an id outside the wire's pattern,
    which the leaf would send as ``q0``: whether the wire refuses the id. Recorded once,
    never overwritten; a 200 means the wire takes the id, and fails the run."""
    if (clef_dir("clef") / f"{QUESTION_ID_PROBE}.json").exists():
        return
    await _raw(QUESTION_ID_PROBE, QUESTION_ID_BODY, QUESTION_ID_REQUEST, refusal=True, clef=True)


def test_live_clef_question_id_probe() -> None:
    """Clef's question-id probe (``-k question_id``): one call, sent only while its
    cassette is absent."""
    asyncio.run(_question_id_probe())


def test_live_clairic() -> None:
    asyncio.run(_clairic_run())


def test_live_clairic_refusals() -> None:
    """The field-name probe and the past-context call, last of Clairic's run: a stop loses
    nothing. Each is recorded once; with both cassettes present it sends nothing."""
    asyncio.run(_refusals())


def test_live_clairic_alias() -> None:
    result = asyncio.run(_clairic_call(ALIAS, aliased(CLAIRIC_PROVIDER)))
    assert isinstance(result, WireResponse), result


def test_live_clairic_order() -> None:
    asyncio.run(_order(CLAIRIC_DIR, _clairic, CLAIRIC_PROVIDER))


def test_live_clairic_extraction() -> None:
    """The extraction over every value type and a list. Its first recording sent a list's
    ``items`` as an object, which the wire refused with 400 and code 5006; the leaf now sends
    the type string, so no replay matches that cassette, ``list-items-object``, and it is
    kept as the record of the refusal."""
    result = asyncio.run(_clairic_call("extraction", EXTRACTION, spans=True))
    assert isinstance(result, WireResponse), result
