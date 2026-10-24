"""The conformance suite, run against the scripted leaf, the dispatcher and the composites,
and shown failing exactly the check a dishonest leaf targets."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import ast
import asyncio
import importlib.util
import pkgutil
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Literal

import pytest

import idia.providers
from idia import (
    UNBOUNDED,
    Capabilities,
    Deadline,
    Dispatcher,
    ExtractionAnswer,
    Fallback,
    Fault,
    Kind,
    Leaf,
    MappedAnswer,
    Provider,
    ProviderError,
    Question,
    Refusal,
    Request,
    RequestFailed,
    RetryPolicy,
    Router,
    Rule,
    RuleFailure,
    Usage,
    WireRequest,
    WireResponse,
)
from idia.conformance import (
    CandidateShape,
    Check,
    CheckNotApplicable,
    Exchange,
    Failure,
    LeafHarness,
    Perturbed,
    ProviderHarness,
    RefusalOnWire,
    Stall,
    Success,
    Unreadable,
    Verdict,
    _review,
    leaf_checks,
    provider_checks,
    run_checks,
)
from idia.conformance._base import validate_spy
from idia.conformance._scripted import CAPABILITIES, CASES, STATE, ScriptedHarness
from idia.providers.recorded import Recorded
from idia.providers.scripted import Item, Reply, Scripted, ScriptFunction
from tests import _dishonest, _dishonest_retry
from tests._closable_scripted import ClosableScripted
from tests._complementing_leaf import ComplementingScripted
from tests._conformance import params, run

SCRIPTED_NOT_APPLICABLE = frozenset({"L9", "L12", "L13", "L15", "L24", "L30"})
"""What a script cannot reach: no SDK, no status mapper, no body, no client, no candidate."""

PROVIDER_NOT_APPLICABLE: Mapping[str, frozenset[str]] = {
    "leaf": frozenset({"P14", "P15", "P16", "P17", "P18", "P23"}),
    "router": frozenset({"P6", "P18", "P19", "P23"}),
    "fallback": frozenset({"P6", "P17", "P19", "P23"}),
}
"""What each scripted shape cannot reach; P23 because the scripted record takes no images."""

_ORDER: tuple[Kind, ...] = ("noul", "choice", "score", "extraction", "box")
_ZERO_DELAY = RetryPolicy(base_delay=0.0, jitter=0.0)


def _verdicts(checks: Sequence[Check]) -> dict[str, Verdict]:
    return {verdict.id: verdict for verdict in asyncio.run(run_checks(checks))}


def _ids(verdicts: Mapping[str, Verdict], result: str) -> set[str]:
    return {id for id, verdict in verdicts.items() if verdict.result == result}


# ----------------------------------------------------------------------------------------
# Harnesses
# ----------------------------------------------------------------------------------------


class WireLikeHarness(ScriptedHarness):
    """The scripted leaf with a stand-in for each exchange a script cannot make, so every
    check's driver is exercised: the suite's own paths, not a claim about the scripted
    leaf."""

    def __init__(self, capabilities: Capabilities | None = None) -> None:
        super().__init__(capabilities or replace(CAPABILITIES, returns_refusal=True))

    def _refuse(self, exchange: Exchange) -> None:
        return None

    def _item(self, exchange: Exchange, wire_request: WireRequest) -> Item:
        if isinstance(exchange, Failure):
            status = exchange.status or 400
            return ProviderError(
                f"status {status}",
                is_retryable=status == 429 or status >= 500,
                retry_after=exchange.retry_after,
                status_code=status,
                request_id=exchange.request_id,
            )
        if isinstance(exchange, Unreadable):
            return ProviderError(f"unreadable body: {exchange.which}", status_code=200)
        reply = super()._item(Success(), wire_request)
        assert isinstance(reply, Reply)
        first = wire_request.questions[0].key
        if isinstance(exchange, RefusalOnWire):
            return replace(reply, answers={**reply.answers, first: Refusal()})
        if isinstance(exchange, CandidateShape):
            failure = RuleFailure(rule=Rule.candidate_shape, observed="estimate", bound=None)
            fault = Fault(key=first, failures=(failure,), mapped=None)
            return replace(reply, answers={**reply.answers, first: fault})
        return super()._item(exchange, wire_request)

    def sdk_retries(self) -> int:
        return 0

    async def client_closed(self, *, injected: bool) -> bool:
        return not injected

    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
        return ClosableScripted(
            capabilities=capabilities, script=script, name=self.name, model=self.model
        )


class ScriptedProviders:
    """A provider harness over scripted members, each wrapped in a zero-delay dispatcher:
    one member alone, a router over them by ``routes`` (kind to member index, in mapping
    order), or a fallback over them in order. The request asks the expected kinds."""

    def __init__(
        self,
        shape: Literal["leaf", "router", "fallback"],
        records: Sequence[Capabilities],
        expected: Capabilities,
        harness: type[ScriptedHarness] = ScriptedHarness,
        routes: Mapping[Kind, int] | None = None,
    ) -> None:
        self._shape: Literal["leaf", "router", "fallback"] = shape
        self._routes = dict(routes or {})
        self._records = tuple(records)
        self._expected = expected
        self._harness = harness
        self._harnesses: list[ScriptedHarness] = []
        self._leaves: list[Scripted] = []

    @property
    def shape(self) -> Literal["leaf", "router", "fallback"]:
        return self._shape

    @property
    def members(self) -> int:
        return len(self._records)

    @property
    def expected_capabilities(self) -> Capabilities:
        return self._expected

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        self._harnesses = [
            self._harness(replace(record, **(overrides or {}))) for record in self._records
        ]
        for i, harness in enumerate(self._harnesses):
            harness.name, harness.model = f"member-{i}", f"model-{i}"
        self._leaves = []
        for harness, plays in zip(self._harnesses, exchanges, strict=True):
            leaf = harness.make(plays)
            assert isinstance(leaf, Scripted)
            self._leaves.append(leaf)
        members = [Dispatcher(self._wrap(leaf), retry=_ZERO_DELAY) for leaf in self._leaves]
        if self._shape == "leaf":
            return members[0]
        if self._shape == "router":
            return Router({kind: members[i] for kind, i in self._routes.items()})
        return Fallback(members)

    def _wrap(self, leaf: Scripted) -> Leaf:
        """What each member's dispatcher wraps: the scripted leaf itself."""
        return leaf

    def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
        if self._shape != "leaf":
            raise CheckNotApplicable(f"a {self._shape} has no one leaf to record")
        harness = self._harness(self._records[0])
        harness.name, harness.model = "member-0", "model-0"
        self._harnesses = [harness]
        return harness.make(exchanges)

    def request(self) -> Request:
        kinds = self._expected.kinds
        questions = tuple(CASES[kind][0][0] for kind in _ORDER if kind in kinds)
        return Request(state=STATE, questions=questions, deadline=Deadline.after(30.0))

    def requests_sent(self) -> int:
        return sum(harness.requests_sent() for harness in self._harnesses)

    def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]:
        return tuple(
            tuple((call.wire_request, call.deadline) for call in leaf.calls)
            for leaf in self._leaves
        )

    def member_names(self) -> tuple[tuple[str, str], ...]:
        return tuple((f"member-{i}", f"model-{i}") for i in range(len(self._records)))


def _wrapped(harness: type[ScriptedHarness] = ScriptedHarness) -> ScriptedProviders:
    return ScriptedProviders("leaf", [CAPABILITIES], CAPABILITIES, harness)


def _router() -> ScriptedProviders:
    """The mapping's first member routes score and extraction, which come after noul in
    the request, and declares box, which nothing routes; the members differ in their token
    limit (one unknown, the other truncating silently) and their label cap."""
    first = replace(
        CAPABILITIES,
        kinds=frozenset({"score", "extraction", "box"}),
        truncates_silently=True,
        max_labels=10,
        confidence={},
    )
    second = replace(CAPABILITIES, kinds=frozenset({"noul", "choice"}), max_input_tokens=None)
    expected = replace(
        CAPABILITIES,
        kinds=frozenset({"noul", "choice", "score", "extraction"}),
        truncates_silently=True,
        opt_in_features=frozenset(),
        # No member takes images, so no image field is known.
        max_image_bytes=None,
        max_total_image_bytes=None,
        min_image_side=None,
        image_media_types=None,
        composite=True,
    )
    routes: dict[Kind, int] = {"score": 0, "extraction": 0, "noul": 1, "choice": 1}
    return ScriptedProviders("router", [first, second], expected, routes=routes)


def _fallback() -> ScriptedProviders:
    """Members with a known and an unknown token limit, the known one truncating silently,
    differing confidence formulas for one kind, and kinds that differ by box."""
    first = replace(CAPABILITIES, truncates_silently=True)
    second = replace(
        CAPABILITIES,
        kinds=frozenset({"noul", "choice", "score", "extraction"}),
        max_input_tokens=None,
        confidence={"choice": "margin"},
    )
    expected = replace(
        CAPABILITIES,
        kinds=frozenset({"noul", "choice", "score", "extraction"}),
        truncates_silently=True,
        confidence={"choice": "mixed"},
        opt_in_features=frozenset(),
        # No member takes images, so no image field is known.
        max_image_bytes=None,
        max_total_image_bytes=None,
        min_image_side=None,
        image_media_types=None,
        composite=True,
    )
    return ScriptedProviders("fallback", [first, second], expected)


_PROVIDERS: Mapping[str, Callable[[], ProviderHarness]] = {
    "leaf": _wrapped,
    "router": _router,
    "fallback": _fallback,
}

# ----------------------------------------------------------------------------------------
# The leaf half against the scripted leaf
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("check", params(leaf_checks(ScriptedHarness)))
def test_scripted_leaf_passes_the_leaf_half(check: Check) -> None:
    if check.id in SCRIPTED_NOT_APPLICABLE:
        with pytest.raises(CheckNotApplicable):
            asyncio.run(check.run())
    else:
        asyncio.run(check.run())


def test_scripted_leaf_not_applicable_set_is_exactly_what_a_script_cannot_reach() -> None:
    verdicts = _verdicts(leaf_checks(ScriptedHarness))
    assert list(verdicts) == [f"L{n}" for n in range(1, 32)]
    assert _ids(verdicts, "not_applicable") == SCRIPTED_NOT_APPLICABLE
    assert _ids(verdicts, "fail") == set()


def test_l11_notes_the_exchanges_the_scripted_leaf_cannot_make() -> None:
    [l11] = [check for check in leaf_checks(ScriptedHarness) if check.id == "L11"]
    note = asyncio.run(l11.run())
    assert note is not None
    assert "Unreadable" in note
    assert asyncio.run(l11.run()) == note


def test_leaf_sdk_retries_are_zero() -> None:
    class Retries(WireLikeHarness):
        def sdk_retries(self) -> int:
            return 2

    assert _verdicts(leaf_checks(WireLikeHarness))["L9"].result == "pass"
    failed = _verdicts(leaf_checks(Retries))["L9"]
    assert failed.result == "fail"
    assert failed.reason is not None
    assert "2" in failed.reason


def test_a_harness_that_drives_every_exchange_reaches_every_check() -> None:
    verdicts = _verdicts(leaf_checks(WireLikeHarness))
    assert _ids(verdicts, "fail") == set()
    assert _ids(verdicts, "not_applicable") == set()


def test_the_review_checks_say_which_module_they_read() -> None:
    verdicts = _verdicts(leaf_checks(ScriptedHarness))
    for id in ("L4", "L10", "L21"):
        reason = verdicts[id].reason
        assert reason is not None
        assert "idia.providers.scripted" in reason


@pytest.mark.parametrize(
    ("record", "not_applicable"),
    [
        pytest.param(
            replace(
                CAPABILITIES,
                kinds=frozenset({"noul"}),
                returns_refusal=True,
                snapshot_pinnable=False,
                opt_in_features=frozenset(),
                confidence={},
            ),
            {"L18", "L25"},
            id="noul-only",
        ),
        pytest.param(
            replace(CAPABILITIES, kinds=frozenset({"choice", "extraction"})),
            {"L20"},
            id="no-noul",
        ),
    ],
)
def test_scripted_leaf_under_other_records(record: Capabilities, not_applicable: set[str]) -> None:
    verdicts = _verdicts(leaf_checks(lambda: ScriptedHarness(record)))
    assert _ids(verdicts, "fail") == set()
    assert _ids(verdicts, "not_applicable") == SCRIPTED_NOT_APPLICABLE | not_applicable


# ----------------------------------------------------------------------------------------
# The provider half against the dispatcher and the composites
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "check"),
    [
        pytest.param(shape, check, id=f"{shape}-{check.id}")
        for shape, make in _PROVIDERS.items()
        for check in provider_checks(make)
    ],
)
def test_providers_pass_the_provider_half(shape: str, check: Check) -> None:
    if check.id in PROVIDER_NOT_APPLICABLE[shape]:
        with pytest.raises(CheckNotApplicable):
            asyncio.run(check.run())
    else:
        run(check)


@pytest.mark.parametrize("shape", list(_PROVIDERS))
def test_provider_not_applicable_set_is_exact(shape: str) -> None:
    verdicts = _verdicts(provider_checks(_PROVIDERS[shape]))
    assert list(verdicts) == [f"P{n}" for n in range(1, 24)]
    assert _ids(verdicts, "fail") == set()
    assert _ids(verdicts, "not_applicable") == PROVIDER_NOT_APPLICABLE[shape]
    reason = verdicts["P12"].reason
    assert reason is not None
    assert reason.startswith("the unreadable-body half was not run")


def test_p19_fails_a_replay_that_does_not_reproduce_the_recording() -> None:
    class Renamed(ScriptedProviders):
        """Replays under another leaf's name, so the recording is not this leaf's."""

        def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
            built = super().leaf(exchanges)
            assert isinstance(built, Scripted)
            if not exchanges:
                built.name = "someone-else"
            return built

    verdicts = _verdicts(provider_checks(lambda: Renamed("leaf", [CAPABILITIES], CAPABILITIES)))
    p19 = verdicts["P19"]
    assert p19.result == "fail"
    assert p19.reason is not None
    assert "recorded from member-0/model-0 with features [], not someone-else/model-0" in (
        p19.reason
    )


def test_p19_compares_errors_by_their_fields() -> None:
    from idia.conformance._provider import _replayable

    assert _replayable(ValueError("x")) == ("ValueError", "x")
    assert _replayable(ValueError("x")) != _replayable(ValueError("y"))


class CredentialFailureHarness(ScriptedHarness):
    """The scripted wire whose recorded failure is a credential error: a 401, not
    retryable."""

    def _item(self, exchange: Exchange, wire_request: WireRequest) -> Item:
        if isinstance(exchange, Failure):
            return ProviderError("credentials refused", status_code=401)
        return super()._item(exchange, wire_request)


def _split_by_credential_error() -> ScriptedProviders:
    record = replace(CAPABILITIES, max_questions=1)
    return ScriptedProviders("leaf", [record], record, CredentialFailureHarness)


def _p11(make: Callable[[], ProviderHarness]) -> Check:
    [p11] = [check for check in provider_checks(make) if check.id == "P11"]
    return p11


def test_p11_passes_outcomes_left_unsent_by_a_credential_error() -> None:
    [verdict] = asyncio.run(run_checks([_p11(_split_by_credential_error)]))
    assert verdict.result == "pass"

    # The same scenario P11 runs reaches the unsent path.
    harness = _split_by_credential_error()
    request = harness.request()
    provider = harness.make([[Failure()] * (len(request.questions) + 2)])
    with pytest.raises(RequestFailed) as raised:
        asyncio.run(provider.answer(request))
    first, *unsent = raised.value.response.outcomes
    assert first.attempts == 1
    assert unsent
    for outcome in unsent:
        assert outcome.attempts == 0
        assert outcome.request_id is None
        assert outcome.error is first.error


class ComplementingHarness(ScriptedHarness):
    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
        return ComplementingScripted(
            capabilities=capabilities, script=script, name=self.name, model=self.model
        )


def test_l21_fails_a_leaf_that_calls_noul_complement() -> None:
    verdicts = _verdicts(leaf_checks(ComplementingHarness))
    assert _ids(verdicts, "fail") == {"L21"}
    reason = verdicts["L21"].reason or ""
    assert "uses noul_complement" in reason
    assert "imports noul_complement" in reason


class AsksAgainProviders(ScriptedProviders):
    """The scripted leaf wrapped in a provider that sends again after no request answered."""

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        return _dishonest.AsksAgain(super().make(exchanges, overrides=overrides))


def test_p20_and_p21_fail_a_provider_that_sends_past_a_credential_error() -> None:
    verdicts = _verdicts(
        provider_checks(lambda: AsksAgainProviders("leaf", [CAPABILITIES], CAPABILITIES))
    )
    assert _ids(verdicts, "fail") == {"P20", "P21"}
    for id in ("P20", "P21"):
        reason = verdicts[id].reason
        assert reason is not None
        assert "each makes exactly its first" in reason


class InventedUnsentProviders(ScriptedProviders):
    """The scripted leaf wrapped in a provider that invents an unsent credential outcome."""

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        return _dishonest.InventsAnUnsentOutcome(super().make(exchanges, overrides=overrides))


def test_p11_fails_an_unsent_outcome_whose_credential_error_no_earlier_outcome_holds() -> None:
    verdicts = _verdicts(
        provider_checks(lambda: InventedUnsentProviders("leaf", [CAPABILITIES], CAPABILITIES))
    )
    # Every row that reads a RequestFailed's last outcome catches the invented one.
    assert _ids(verdicts, "fail") == {"P11", "P20", "P21", "P22"}
    assert _ids(verdicts, "not_applicable") == PROVIDER_NOT_APPLICABLE["leaf"]
    reason = verdicts["P11"].reason
    assert reason is not None
    assert "lists no failed attempts" in reason


class RecordedProviders(ScriptedProviders):
    """The scripted leaf under a recorded leaf recording to a fresh log, wrapped."""

    def __init__(self, directory: Path) -> None:
        super().__init__("leaf", [CAPABILITIES], CAPABILITIES)
        self._directory = directory

    def _wrap(self, leaf: Scripted) -> Leaf:
        return Recorded(leaf, self._directory / f"{uuid.uuid4().hex}.json", "once")


def test_the_recorded_leaf_wrapped_passes_the_provider_half(tmp_path: Path) -> None:
    verdicts = _verdicts(provider_checks(lambda: RecordedProviders(tmp_path)))
    assert _ids(verdicts, "fail") == set()
    assert _ids(verdicts, "not_applicable") == {"P14", "P15", "P16", "P17", "P18", "P23"}


class _OwnRulesProvider(ScriptedProviders):
    """A dispatcher over a leaf that validates by its own rules, never calling validate."""

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        harness = _dishonest.OwnRulesHarness(replace(self._records[0], **(overrides or {})))
        self._harnesses = [harness]
        return Dispatcher(harness.make(exchanges[0]), retry=_ZERO_DELAY)


def test_p13_fails_a_provider_whose_answers_never_pass_validate() -> None:
    checks = provider_checks(lambda: _OwnRulesProvider("leaf", [CAPABILITIES], CAPABILITIES))
    [p13] = [check for check in checks if check.id == "P13"]
    with pytest.raises(AssertionError, match="without passing validate"):
        asyncio.run(p13.run())


def test_unreadable_body_half_of_p12_runs_where_the_harness_can_make_one() -> None:
    verdicts = _verdicts(provider_checks(lambda: _wrapped(WireLikeHarness)))
    assert verdicts["P12"] == Verdict("P12", "pass")
    assert _ids(verdicts, "fail") == set()


class UnperturbableProviders(ScriptedProviders):
    """A wrapped scripted leaf whose harness can make neither a moved sum nor an
    unreadable body, as a chat leaf's renormalized read over a transport with no body."""

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        if any(isinstance(e, Perturbed | Unreadable) for plays in exchanges for e in plays):
            raise CheckNotApplicable("this harness makes no such exchange")
        return super().make(exchanges, overrides=overrides)


def test_p12_runs_whichever_half_the_harness_can_make() -> None:
    verdicts = _verdicts(
        provider_checks(lambda: UnperturbableProviders("leaf", [CAPABILITIES], CAPABILITIES))
    )
    p12 = verdicts["P12"]
    assert p12.result == "not_applicable"
    assert p12.reason is not None
    assert "the fault half was not run" in p12.reason
    assert "the unreadable-body half was not run" in p12.reason


# ----------------------------------------------------------------------------------------
# Dishonest leaves fail exactly the check they target
# ----------------------------------------------------------------------------------------


def _wrong_box() -> ScriptedHarness:
    """Claims box, but its wire answers a box question with a noul's shape."""
    cases = dict(CASES)
    cases["box"] = tuple((question, CASES["noul"][0][1]) for question, _ in CASES["box"])
    return ScriptedHarness(cases=cases)


def _refusing() -> ScriptedHarness:
    """Returns a Refusal while its record says returns_refusal is false."""
    cases = dict(CASES)
    cases["choice"] = (CASES["choice"][0], (CASES["choice"][1][0], Refusal()))
    return ScriptedHarness(cases=cases)


@pytest.mark.parametrize(
    ("make_harness", "target"),
    [
        pytest.param(_dishonest_retry.RetryingHarness, {"L8", "L10"}, id="retries-inside-call"),
        pytest.param(_dishonest.OwnRulesHarness, {"L16"}, id="skips-validate"),
        pytest.param(_wrong_box, {"L5", "L17"}, id="claims-a-kind-it-cannot-answer"),
        pytest.param(_refusing, {"L23"}, id="refuses-without-returns-refusal"),
    ],
)
def test_dishonest_leaf_fails_exactly_its_target(
    make_harness: Callable[[], LeafHarness], target: set[str]
) -> None:
    verdicts = _verdicts(leaf_checks(make_harness))
    assert _ids(verdicts, "fail") == target
    assert _ids(verdicts, "not_applicable") == SCRIPTED_NOT_APPLICABLE


_SOURCES: Mapping[str, tuple[set[str], str]] = {
    "complement": (
        {"L21"},
        """
class Leaf(Scripted):
    def _yes(self, p_no: float) -> float:
        return 1 - p_no
""",
    ),
    "class-limit": (
        {"L4"},
        """
class Leaf(Scripted):
    MAX_INPUT_TOKENS = 4096
""",
    ),
    "base-class-limit": (
        {"L4"},
        """
class _Base(Scripted):
    max_labels = 3

class Leaf(_Base):
    pass
""",
    ),
    "wire-model-beside-the-leaf": (
        set(),
        """
class _Wire:
    confidence: float | None = None

class Leaf(Scripted):
    pass
""",
    ),
    "swallows-a-provider-error": (
        {"L10"},
        """
from idia import ProviderError


class Leaf(Scripted):
    async def call(self, wire_request, deadline):
        try:
            return await super().call(wire_request, deadline)
        except ProviderError as error:
            caught = error
        raise caught
""",
    ),
    "re-raises-inside-a-loop": (
        {"L10"},
        """
from idia import ProviderError


class Leaf(Scripted):
    async def call(self, wire_request, deadline):
        for _ in range(1):
            try:
                return await super().call(wire_request, deadline)
            except ProviderError:
                raise
""",
    ),
    "returns-from-a-provider-error": (
        {"L10"},
        """
from idia import ProviderError


class Leaf(Scripted):
    async def call(self, wire_request, deadline):
        try:
            return await super().call(wire_request, deadline)
        except ProviderError as error:
            if error.status_code == 418:
                return None
            raise
""",
    ),
    "passes-a-provider-error-on": (
        set(),
        """
from idia import ProviderError


class Leaf(Scripted):
    async def call(self, wire_request, deadline):
        try:
            return await super().call(wire_request, deadline)
        except ProviderError:
            def note():
                return "seen"

            self.seen = note()
            raise
""",
    ),
}


@pytest.mark.parametrize("name", list(_SOURCES))
def test_dishonest_leaf_source_fails_exactly_its_review_check(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, source = _SOURCES[name]
    module_name = f"dishonest_{name.replace('-', '_')}"
    path = tmp_path / f"{module_name}.py"
    header = "from idia.providers.scripted import Scripted\n"
    path.write_text(header + source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    leaf_type: type[Scripted] = module.Leaf

    class Harness(ScriptedHarness):
        def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
            return leaf_type(capabilities=capabilities, script=script)

    verdicts = _verdicts(leaf_checks(Harness))
    assert _ids(verdicts, "fail") == target


def test_retrying_leaf_is_caught_by_the_request_count() -> None:
    [l8] = [check for check in leaf_checks(_dishonest_retry.RetryingHarness) if check.id == "L8"]
    with pytest.raises(AssertionError, match="sent 2 wire requests"):
        asyncio.run(l8.run())


# ----------------------------------------------------------------------------------------
# The review walks
# ----------------------------------------------------------------------------------------


def _tree(source: str) -> ast.Module:
    return ast.parse(source)


def _imports_under(tree: ast.Module, module_name: str, package: str) -> list[str]:
    """Each import of ``package`` or a module under it."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            sources = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            source = _review._resolve(node, module_name)
            sources = [source, *(f"{source}.{alias.name}" for alias in node.names)]
        else:
            continue
        if any(name == package or name.startswith(package + ".") for name in sources):
            found.append(f"line {node.lineno}: {ast.unparse(node)}")
    return found


def _validation_imports(tree: ast.Module, module_name: str) -> list[str]:
    """Each import of ``validate``, ``validate_answers`` or the module holding them."""
    found = _imports_under(tree, module_name, "idia.answers")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _review._resolve(node, module_name) == "idia":
            names = {alias.name for alias in node.names}
            if names & {"validate", "validate_answers"}:
                found.append(f"line {node.lineno}: {ast.unparse(node)}")
    return found


def test_core_imports_nothing_from_providers() -> None:
    tree = _review.parse("idia.core")
    assert _imports_under(tree, "idia.core", "idia.providers") == []


def test_no_core_path_imports_validation() -> None:
    for module in ("idia.dispatch", "idia.fit"):
        assert _validation_imports(_review.parse(module), module) == []


def test_no_module_under_providers_reaches_into_the_call_loop() -> None:
    extras = {module for modules in idia.providers._EXTRAS.values() for module in modules}
    names = ["idia.providers"]
    for module in pkgutil.walk_packages(idia.providers.__path__, "idia.providers."):
        try:
            importlib.import_module(module.name)
        except ModuleNotFoundError as error:
            # An extra the jobs that sync every extra leave out is reviewed by its own job.
            if (error.name or "").split(".")[0] not in extras:
                raise
            continue
        names.append(module.name)
    for name in names:
        tree = _review.parse(name)
        assert _review.loop_imports(tree, name) == []
        assert _review.provider_error_catches(tree, name) == []
        assert _review.class_limits(tree) == []
        assert _review.complements(tree, name) == []


def test_loop_imports_are_found_however_they_are_written() -> None:
    source = """
import idia.dispatch
from idia.retry import RetryPolicy
from ..limits import RateLimiter
from .. import dispatch
from idia import Dispatcher, Question
"""
    found = _review.loop_imports(_tree(source), "idia.providers.wire")
    assert len(found) == 5
    assert "from idia import Question" not in "".join(found)


def test_every_catch_that_could_take_a_provider_error_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = type(sys)("catching")
    module.Alias = ProviderError
    monkeypatch.setitem(sys.modules, "catching", module)
    source = """
try: pass
except: pass
try: pass
except BaseException: pass
try: pass
except (ValueError, core.DeadlineExceeded): pass
try: pass
except Alias: pass
try: pass
except Exception: pass
try: pass
except Exception: raise
try: pass
except (OSError, Exception) as error: raise ProviderError("wrapped") from error
try: pass
except errors(): pass
try: pass
except ProviderError: raise
try: pass
except ProviderError as error: raise error
for item in items:
    try: pass
    except ProviderError: raise
while True:
    def inner():
        try: pass
        except ProviderError: raise
try: pass
except ProviderError: record()
try: pass
except ProviderError:
    if ok(): return None
    raise
try: pass
except ProviderError:
    hook = lambda: None
    def later(): return 1
    raise
"""
    found = _review.provider_error_catches(_tree(source), "catching")
    assert [line.split(": ", 1)[1] for line in found] == [
        "except bare except",
        "except BaseException",
        "except DeadlineExceeded",
        "except Alias",
        "except Exception",
        "except ProviderError",
        "except ProviderError",
        "except ProviderError",
        "except ProviderError",
    ]
    lines = [int(line.split(":")[0].removeprefix("line ")) for line in found[-4:]]
    assert lines == [21, 24, 30, 32]


def test_class_limits_and_complements_are_found() -> None:
    source = """
class Leaf:
    capabilities = make()
    max_labels: int = 3
    name: str
    other = 1

def reduce(p):
    return 1.0 - p, True - p, 2 - p
"""
    assert _review.class_limits(_tree(source)) == [
        "line 3: Leaf.capabilities",
        "line 4: Leaf.max_labels",
    ]
    assert _review.complements(_tree(source), "m") == ["line 9: 1.0 - p"]


def test_complement_allowlist_admits_a_named_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_review, "_COMPLEMENT_ALLOWED", frozenset({"chat:Leaf.reduce"}))
    source = "class Leaf:\n    def reduce(self, p):\n        return 1 - p\n"
    assert _review.complements(_tree(source), "chat") == []
    assert _review.complements(_tree(source), "other") == ["line 3: 1 - p"]


def test_validation_imports_are_found() -> None:
    source = """
import idia.answers
from idia.answers import validate
from idia import validate_answers
from . import answers
from idia import Question
"""
    found = _validation_imports(_tree(source), "idia.fit")
    assert len(found) == 4


def test_a_module_with_no_source_is_not_applicable() -> None:
    with pytest.raises(CheckNotApplicable, match="cannot be read"):
        _review.parse("sys")
    with pytest.raises(CheckNotApplicable, match="cannot be read"):
        _review.parse("not.loaded.anywhere")


# ----------------------------------------------------------------------------------------
# Verdicts and the spy
# ----------------------------------------------------------------------------------------


def test_run_checks_reports_any_exception_as_a_failure() -> None:
    async def broken() -> str | None:
        raise ValueError("the harness broke")

    async def passing() -> str | None:
        return "a note"

    verdicts = asyncio.run(
        run_checks([Check("X1", "broken", broken), Check("X2", "passing", passing)])
    )
    assert verdicts == (
        Verdict("X1", "fail", "ValueError: the harness broke"),
        Verdict("X2", "pass", "a note"),
    )


def test_validate_spy_patches_every_binding_and_restores_it() -> None:
    import idia
    import idia.answers

    real = idia.answers.validate
    with validate_spy() as validated:
        assert idia.answers.validate is not real
        assert idia.validate is idia.answers.validate
        idia.validate_answers((CASES["noul"][0][0],), {}, CAPABILITIES.rounding)
        idia.answers.validate(CASES["noul"][0][0], CASES["noul"][0][1], CAPABILITIES.rounding)
    assert validated == [("noul_a", CAPABILITIES.rounding)]
    assert idia.answers.validate is real
    assert idia.validate is real


def test_scripted_harness_refuses_a_perturbation_it_cannot_make() -> None:
    harness = ScriptedHarness()
    with pytest.raises(ValueError, match="cannot perturb"):
        harness._item(Perturbed("noul", "sum"), harness.wire_request("noul"))


def test_scripted_harness_runs_out_like_a_script() -> None:
    harness = ScriptedHarness()
    leaf = harness.make([])
    with pytest.raises(AssertionError, match="ran out"):
        asyncio.run(leaf.call(harness.wire_request("noul"), Deadline.after(5.0)))


# ----------------------------------------------------------------------------------------
# Records that take images
# ----------------------------------------------------------------------------------------

_IMAGES = replace(
    CAPABILITIES,
    modalities=frozenset({"text", "image"}),
    image_carriers=frozenset({"parts"}),
    max_images=UNBOUNDED,
    max_images_per_question=UNBOUNDED,
    max_image_bytes=4096,
    max_total_image_bytes=8192,
    min_image_side=16,
    image_media_types=frozenset({"image/png", "image/jpeg"}),
)
"""The scripted record taking images, every image field known."""

_IMAGE_ROUTES: Mapping[Kind, int] = {"score": 0, "extraction": 0, "noul": 1, "choice": 1}


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: ScriptedProviders("leaf", [_IMAGES], _IMAGES), id="leaf"),
        pytest.param(
            lambda: ScriptedProviders(
                "router",
                [_IMAGES, _IMAGES],
                replace(_IMAGES, kinds=frozenset(_IMAGE_ROUTES)),
                routes=_IMAGE_ROUTES,
            ),
            id="router",
        ),
        pytest.param(
            lambda: ScriptedProviders("fallback", [_IMAGES, _IMAGES], _IMAGES), id="fallback"
        ),
    ],
)
def test_p23_refuses_each_image_past_the_record_on_every_shape(
    make: Callable[[], ProviderHarness],
) -> None:
    [p23] = [check for check in provider_checks(make) if check.id == "P23"]
    [verdict] = asyncio.run(run_checks([p23]))
    assert verdict == Verdict("P23", "pass")


class SkipsImageChecks(ScriptedProviders):
    """Members whose records know no image field, so an image past the declared record
    reaches the leaf: a provider that sends what it should refuse."""

    def make(
        self,
        exchanges: Sequence[Sequence[Exchange]],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> Provider:
        unchecked = {
            **(overrides or {}),
            "max_image_bytes": UNBOUNDED,
            "max_total_image_bytes": UNBOUNDED,
            "min_image_side": 0,
            "image_media_types": None,
        }
        return super().make(exchanges, overrides=unchecked)


def test_p23_fails_a_provider_that_sends_an_image_past_its_record() -> None:
    verdicts = _verdicts(provider_checks(lambda: SkipsImageChecks("leaf", [_IMAGES], _IMAGES)))
    assert _ids(verdicts, "fail") == {"P23"}


# A leaf taking images with a byte cap of 0 needs no L31 double: Core's constructor
# refuses the record (tests/test_core.py).


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(
            {
                "max_image_bytes": None,
                "max_total_image_bytes": None,
                "min_image_side": None,
                "image_media_types": None,
            },
            id="all-unknown",
        ),
        pytest.param({"max_total_image_bytes": None, "image_media_types": None}, id="mixed"),
    ],
)
def test_l31_passes_a_leaf_without_images_that_leaves_image_fields_unknown(
    fields: dict[str, object],
) -> None:
    verdicts = _verdicts(leaf_checks(lambda: ScriptedHarness(replace(CAPABILITIES, **fields))))
    assert verdicts["L31"].result == "pass"


def test_l31_fails_a_leaf_without_images_that_declares_an_unbounded_image_cap() -> None:
    stray = replace(CAPABILITIES, max_image_bytes=UNBOUNDED)
    verdicts = _verdicts(leaf_checks(lambda: ScriptedHarness(stray)))
    assert _ids(verdicts, "fail") == {"L31"}


def test_l31_fails_a_leaf_without_images_that_declares_image_fields() -> None:
    stray = replace(CAPABILITIES, max_image_bytes=4096)
    verdicts = _verdicts(leaf_checks(lambda: ScriptedHarness(stray)))
    assert _ids(verdicts, "fail") == {"L31"}


@pytest.mark.parametrize("first", [0, 1], ids=["image-member-first", "text-member-first"])
def test_p23_passes_a_router_whose_text_member_refuses_the_state_on_its_modalities(
    first: int,
) -> None:
    records = [_IMAGES, CAPABILITIES] if first == 0 else [CAPABILITIES, _IMAGES]

    def make() -> ProviderHarness:
        expected = replace(_IMAGES, kinds=frozenset(_IMAGE_ROUTES))
        return ScriptedProviders("router", records, expected, routes=_IMAGE_ROUTES)

    [p23] = [check for check in provider_checks(make) if check.id == "P23"]
    [verdict] = asyncio.run(run_checks([p23]))
    assert verdict == Verdict("P23", "pass")


def test_p23_builds_jpeg_images_for_a_record_that_takes_jpeg_alone() -> None:
    jpeg = replace(_IMAGES, image_media_types=frozenset({"image/jpeg"}))
    [p23] = [
        check
        for check in provider_checks(lambda: ScriptedProviders("leaf", [jpeg], jpeg))
        if check.id == "P23"
    ]
    [verdict] = asyncio.run(run_checks([p23]))
    assert verdict == Verdict("P23", "pass")


def test_l31_fails_a_leaf_that_takes_images_by_two_carriers() -> None:
    two = replace(_IMAGES, image_carriers=frozenset({"parts", "array"}))
    verdicts = _verdicts(leaf_checks(lambda: ScriptedHarness(two)))
    assert _ids(verdicts, "fail") == {"L31"}


class ImagesFeatureHarness(ScriptedHarness):
    """A scripted leaf whose ``images`` feature adds the image modality; with it off, the
    record states the image fields the feature brings. ``drift`` makes them differ."""

    drift: bool = False

    def __init__(self) -> None:
        super().__init__(
            replace(
                _IMAGES,
                modalities=frozenset({"text"}),
                image_carriers=frozenset(),
                max_images=0,
                max_images_per_question=0,
                opt_in_features=frozenset({"images"}),
                feature_modalities={"images": frozenset({"image"})},
            )
        )

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        if "images" not in features:
            return super().capabilities_for(features)
        return replace(
            self.capabilities,
            enabled_features=features,
            modalities=frozenset({"text", "image"}),
            image_carriers=frozenset({"parts"}),
            max_images=UNBOUNDED,
            max_images_per_question=UNBOUNDED,
            max_image_bytes=2048 if self.drift else self.capabilities.max_image_bytes,
        )


class DriftingImagesHarness(ImagesFeatureHarness):
    drift = True


def test_l31_passes_a_leaf_whose_image_fields_off_are_those_its_feature_brings() -> None:
    verdicts = _verdicts(leaf_checks(ImagesFeatureHarness))
    assert verdicts["L31"].result == "pass"
    drifting = _verdicts(leaf_checks(DriftingImagesHarness))
    assert _ids(drifting, "fail") == {"L31"}
    assert "not those the images feature would bring" in (drifting["L31"].reason or "")


# ----------------------------------------------------------------------------------------
# Records that leave a check nothing to drive
# ----------------------------------------------------------------------------------------

_OPEN = replace(
    CAPABILITIES,
    modalities=frozenset({"text", "image"}),
    image_carriers=frozenset({"parts"}),
    max_labels=UNBOUNDED,
    max_levels=UNBOUNDED,
    max_input_tokens=UNBOUNDED,
    max_images=UNBOUNDED,
    max_images_per_question=UNBOUNDED,
    max_image_bytes=UNBOUNDED,
    max_total_image_bytes=UNBOUNDED,
    image_media_types=None,
)
"""Every kind, images too, no cap and no known size limit: nothing in it to exceed."""


@pytest.mark.parametrize(
    ("make", "not_applicable"),
    [
        pytest.param(
            lambda: ScriptedProviders("leaf", [_OPEN], _OPEN),
            {"P3", "P6"},
            id="leaf-nothing-to-exceed",
        ),
        pytest.param(
            lambda: ScriptedProviders(
                "fallback",
                [_OPEN, _OPEN],
                replace(_OPEN, opt_in_features=frozenset(), composite=True),
            ),
            {"P3", "P6", "P16"},
            id="fallback-nothing-to-exceed",
        ),
        pytest.param(
            lambda: _narrow(frozenset({"noul", "extraction"})), {"P12"}, id="no-sum-to-move"
        ),
        pytest.param(
            lambda: _narrow(frozenset({"noul", "choice", "score"})), set[str](), id="no-extraction"
        ),
        pytest.param(lambda: _narrow(frozenset({"choice", "box"})), set[str](), id="no-noul"),
        pytest.param(lambda: _featured(_FEATURED), set[str](), id="an-off-feature-adds-them"),
    ],
)
def test_provider_half_under_other_records(
    make: Callable[[], ProviderHarness], not_applicable: set[str]
) -> None:
    harness = make()
    verdicts = _verdicts(provider_checks(make))
    assert _ids(verdicts, "fail") == set()
    expected = PROVIDER_NOT_APPLICABLE[harness.shape] | not_applicable
    assert _ids(verdicts, "not_applicable") == expected


def _narrow(kinds: frozenset[Kind]) -> ScriptedProviders:
    record = replace(CAPABILITIES, kinds=kinds)
    return ScriptedProviders("leaf", [record], record)


_FEATURED = replace(
    CAPABILITIES,
    kinds=frozenset({"noul", "choice", "score"}),
    feature_kinds={"long-context": frozenset({"extraction", "box"})},
    feature_modalities={"long-context": frozenset({"image"})},
)
"""An off feature that adds the kinds and the modality the record lacks, so P6 expects it
named for those misfits as well as for the caps."""


def _featured(expected: Capabilities, leaf: Capabilities = _FEATURED) -> ScriptedProviders:
    return ScriptedProviders("leaf", [leaf], expected)


def test_p6_fails_a_reason_that_does_not_name_the_feature_that_adds_the_missing_kind() -> None:
    # The leaf's own record says nothing adds the missing kind; the expected record says
    # long-context does, so the reason names too few off features.
    bare = replace(_FEATURED, feature_kinds={}, feature_modalities={})
    [p6] = [
        check for check in provider_checks(lambda: _featured(_FEATURED, bare)) if check.id == "P6"
    ]
    with pytest.raises(AssertionError, match=r"names off features \[\], not \['long-context'\]"):
        asyncio.run(p6.run())


def test_p18_is_not_applicable_when_the_harness_cannot_see_deadlines() -> None:
    class Blind(ScriptedProviders):
        def member_calls(self) -> tuple[tuple[tuple[WireRequest, Deadline | None], ...], ...]:
            return tuple(
                tuple((wire_request, None) for wire_request, _ in log)
                for log in super().member_calls()
            )

    def make() -> ProviderHarness:
        fallback = _fallback()
        return Blind("fallback", fallback._records, fallback.expected_capabilities)

    [p18] = [check for check in provider_checks(make) if check.id == "P18"]
    with pytest.raises(CheckNotApplicable, match="cannot see the deadline"):
        asyncio.run(p18.run())


def test_a_record_that_can_be_assigned_to_fails_l3() -> None:
    class Loose(Capabilities):
        __setattr__ = object.__setattr__

    values = {field.name: getattr(CAPABILITIES, field.name) for field in fields(Capabilities)}
    loose = Loose(**values)
    [l3] = [check for check in leaf_checks(lambda: ScriptedHarness(loose)) if check.id == "L3"]
    with pytest.raises(AssertionError, match="did not raise"):
        asyncio.run(l3.run())


def test_imports_under_a_package_are_found() -> None:
    source = "from idia.providers import scripted\nfrom idia import providers\n"
    found = _imports_under(_tree(source), "idia.core", "idia.providers")
    assert len(found) == 2


def test_relative_imports_resolve_in_a_package_init_and_in_a_subpackage_module() -> None:
    importlib.import_module("idia.conformance._leaf")  # loaded: __package__ is known

    init = _tree("from .scripted import Scripted\nfrom .. import retry\n")
    assert _review.loop_imports(init, "idia.providers") == ["line 2: from idia import retry"]
    nodes = [n for n in ast.walk(init) if isinstance(n, ast.ImportFrom)]
    assert _review._resolve(nodes[0], "idia.providers") == "idia.providers.scripted"
    inner = _tree("from ._base import expect\nfrom ..dispatch import Router\n")
    nodes = [n for n in ast.walk(inner) if isinstance(n, ast.ImportFrom)]
    assert _review._resolve(nodes[0], "idia.conformance._leaf") == ("idia.conformance._base")
    assert _review.loop_imports(inner, "idia.conformance._leaf") == [
        "line 2: from idia.dispatch import ..."
    ]
    beyond = _tree("from .... import x\n")
    [node] = [n for n in ast.walk(beyond) if isinstance(n, ast.ImportFrom)]
    assert _review._resolve(node, "idia.core") == ""


def test_the_suite_imports_only_idia_and_the_standard_library() -> None:
    package = importlib.import_module("idia.conformance")
    names = ["idia.conformance"] + [
        m.name for m in pkgutil.iter_modules(package.__path__, "idia.conformance.")
    ]
    for name in names:
        importlib.import_module(name)
        for node in ast.walk(_review.parse(name)):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            foreign = [r for r in roots if r != "idia" and r not in sys.stdlib_module_names]
            assert foreign == [], f"{name} imports {foreign}"


def test_l11_is_not_applicable_when_the_harness_can_make_no_failure() -> None:
    class Mute(ScriptedHarness):
        def _refuse(self, exchange: Exchange) -> None:
            raise CheckNotApplicable("this harness makes only successes")

    [l11] = [check for check in leaf_checks(Mute) if check.id == "L11"]
    with pytest.raises(CheckNotApplicable, match="can make no failure"):
        asyncio.run(l11.run())


def test_l7_fails_a_leaf_whose_feature_does_not_change_what_it_should() -> None:
    class Promised(ScriptedHarness):
        def capabilities_for(self, features: frozenset[str]) -> Capabilities:
            record = super().capabilities_for(features)
            return replace(record, max_input_tokens=200_000) if features else record

        def make(
            self, exchanges: Sequence[Exchange], features: frozenset[str] | None = None
        ) -> Leaf:
            built = super().make(exchanges, features)
            assert isinstance(built, Scripted)
            built.capabilities = super().capabilities_for(built.capabilities.enabled_features)
            return built

    [l7] = [check for check in leaf_checks(Promised) if check.id == "L7"]
    with pytest.raises(AssertionError, match="differs in \\['max_input_tokens'\\]"):
        asyncio.run(l7.run())


def test_l7_fails_a_leaf_whose_feature_maps_follow_what_is_enabled() -> None:
    class Following(ScriptedHarness):
        """The leaf and the harness agree, but both map the feature only while it is on."""

        def capabilities_for(self, features: frozenset[str]) -> Capabilities:
            record = super().capabilities_for(features)
            adds = {feature: frozenset[Kind]({"box"}) for feature in features}
            return replace(record, feature_kinds=adds)

    [l7] = [check for check in leaf_checks(Following) if check.id == "L7"]
    with pytest.raises(AssertionError, match="feature_kinds changes"):
        asyncio.run(l7.run())


# ----------------------------------------------------------------------------------------
# A leaf whose runtime has no statuses: L8 and P19 run the drivers it can make
# ----------------------------------------------------------------------------------------


class StatuslessHarness(ScriptedHarness):
    """The scripted leaf standing in for a runtime with no statuses, as a model run in
    process has: a failure with a status cannot be made, and the recorded failure carries
    none."""

    def _refuse(self, exchange: Exchange) -> None:
        if isinstance(exchange, Failure) and exchange.status is not None:
            raise CheckNotApplicable("the runtime has no statuses")
        super()._refuse(exchange)

    def _item(self, exchange: Exchange, wire_request: WireRequest) -> Item:
        if isinstance(exchange, Failure):
            return ProviderError("the runtime failed", is_retryable=False)
        return super()._item(exchange, wire_request)


class RetriesItsCut:
    """Wraps a leaf and, when its deadline cuts a call, calls again under a fresh one."""

    def __init__(self, inner: Leaf) -> None:
        self._inner = inner
        self.name, self.model = inner.name, inner.model
        self.capabilities, self.estimator = inner.capabilities, inner.estimator

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        try:
            return await self._inner.call(wire_request, deadline)
        except ProviderError as cut:
            if not cut.is_retryable:
                raise
            return await self._inner.call(wire_request, Deadline.after(5.0))


class RetryingStatuslessHarness(StatuslessHarness):
    """A statusless leaf that calls again when its deadline cuts the first call."""

    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
        return RetriesItsCut(super()._build(capabilities, script))


class SendsTwice:
    """Wraps a leaf and sends every wire request a second time after an answer."""

    def __init__(self, inner: Leaf) -> None:
        self._inner = inner
        self.name, self.model = inner.name, inner.model
        self.capabilities, self.estimator = inner.capabilities, inner.estimator

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        await self._inner.call(wire_request, deadline)
        return await self._inner.call(wire_request, deadline)


class SendsTwiceHarness(ScriptedHarness):
    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
        return SendsTwice(super()._build(capabilities, script))


class StatuslessSendsTwiceHarness(StatuslessHarness):
    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
        return SendsTwice(super()._build(capabilities, script))


class NothingHarness(StatuslessHarness):
    """Can make neither a success nor a stall, so neither half of L8 runs."""

    def _refuse(self, exchange: Exchange) -> None:
        if isinstance(exchange, Success | Stall):
            raise CheckNotApplicable(f"cannot make {exchange}")
        super()._refuse(exchange)


def _check(checks: Sequence[Check], id: str) -> Verdict:
    [verdict] = asyncio.run(run_checks([check for check in checks if check.id == id]))
    return verdict


def test_l8_runs_its_retryable_half_over_a_stall_where_no_429_can_be_made() -> None:
    l8 = _check(leaf_checks(StatuslessHarness), "L8")
    assert l8.result == "pass"
    assert l8.reason == (
        "the retryable half ran over a stall, not a 429: the runtime has no statuses"
    )


def test_statusless_leaf_not_applicable_set_is_exact() -> None:
    verdicts = _verdicts(leaf_checks(StatuslessHarness))
    assert _ids(verdicts, "fail") == set()
    assert _ids(verdicts, "not_applicable") == SCRIPTED_NOT_APPLICABLE


def test_l8_fails_a_leaf_that_calls_again_after_its_deadline_cut() -> None:
    l8 = _check(leaf_checks(RetryingStatuslessHarness), "L8")
    assert l8.result == "fail"
    assert l8.reason is not None
    assert "one call cut by its deadline sent 2 wire requests" in l8.reason


def test_l8_fails_a_leaf_that_sends_twice_over_a_success() -> None:
    l8 = _check(leaf_checks(StatuslessSendsTwiceHarness), "L8")
    assert l8.result == "fail"
    assert l8.reason is not None
    assert "one call over a success sent 2 wire requests" in l8.reason


def test_l8_fails_a_leaf_whose_call_after_a_429_sends_twice() -> None:
    l8 = _check(leaf_checks(SendsTwiceHarness), "L8")
    assert l8.result == "fail"
    assert l8.reason is not None
    assert "the call after a retryable failure sent 2 wire requests" in l8.reason


def test_l8_is_not_applicable_only_when_neither_half_runs() -> None:
    l8 = _check(leaf_checks(NothingHarness), "L8")
    assert l8.result == "not_applicable"
    assert l8.reason is not None
    assert "the retryable half was not run" in l8.reason
    assert "the success half was not run" in l8.reason


def test_p19_records_each_log_a_statusless_leaf_can_make_and_notes_the_one_it_skips() -> None:
    p19 = _check(provider_checks(lambda: _wrapped(StatuslessHarness)), "P19")
    assert p19 == Verdict(
        "P19", "pass", "a 503 then an answer was not recorded: the runtime has no statuses"
    )


def test_p19_fails_a_statusless_replay_that_does_not_reproduce_the_recording() -> None:
    class Renamed(ScriptedProviders):
        def leaf(self, exchanges: Sequence[Exchange]) -> Leaf:
            built = super().leaf(exchanges)
            assert isinstance(built, Scripted)
            if not exchanges:
                built.name = "someone-else"
            return built

    p19 = _check(
        provider_checks(lambda: Renamed("leaf", [CAPABILITIES], CAPABILITIES, StatuslessHarness)),
        "P19",
    )
    assert p19.result == "fail"
    assert p19.reason is not None
    assert "not someone-else/model-0" in p19.reason


# ----------------------------------------------------------------------------------------
# L25 for a leaf whose model gives no guess
# ----------------------------------------------------------------------------------------


def _no_guess_cases(
    answer: ExtractionAnswer,
) -> dict[Kind, tuple[tuple[Question, MappedAnswer], ...]]:
    cases = dict(CASES)
    cases["extraction"] = tuple((question, answer) for question, _ in CASES["extraction"])
    return cases


def test_l25_checks_only_that_no_guess_is_filled_where_no_case_has_one() -> None:
    harness = ScriptedHarness(cases=_no_guess_cases(ExtractionAnswer(value="Ada")))
    l25 = _check(leaf_checks(lambda: harness), "L25")
    assert l25.result == "pass"
    assert l25.reason is not None
    assert l25.reason.startswith("no expected extraction has a null value and a guess")


def test_l25_fails_a_leaf_that_fills_a_guess_where_the_model_gave_none() -> None:
    class FillsAGuess(ScriptedHarness):
        def _item(self, exchange: Exchange, wire_request: WireRequest) -> Item:
            reply = super()._item(exchange, wire_request)
            if not isinstance(reply, Reply):
                return reply
            filled = {
                key: replace(answer, guess="Eve")
                if isinstance(answer, ExtractionAnswer)
                else answer
                for key, answer in reply.answers.items()
            }
            return replace(reply, answers=filled)

    harness = FillsAGuess(cases=_no_guess_cases(ExtractionAnswer(value="Ada")))
    l25 = _check(leaf_checks(lambda: harness), "L25")
    assert l25.result == "fail"
    assert l25.reason is not None
    assert "guess None became" in l25.reason


def test_l25_is_not_applicable_where_no_extraction_answer_is_expected() -> None:
    cases = dict(CASES)
    cases["extraction"] = tuple((question, Refusal()) for question, _ in CASES["extraction"])
    harness = ScriptedHarness(replace(CAPABILITIES, returns_refusal=True), cases=cases)
    l25 = _check(leaf_checks(lambda: harness), "L25")
    assert l25 == Verdict("L25", "not_applicable", "the harness expects no extraction answer")


# ----------------------------------------------------------------------------------------
# L26 for a leaf whose wire or runtime gives no request id or output count
# ----------------------------------------------------------------------------------------


class Strips:
    """Wraps a leaf and replaces fields of each wire response it returns."""

    def __init__(self, inner: Leaf, **fields: object) -> None:
        self._inner = inner
        self._fields = fields
        self.name, self.model = inner.name, inner.model
        self.capabilities, self.estimator = inner.capabilities, inner.estimator

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        response = await self._inner.call(wire_request, deadline)
        return replace(response, **self._fields)  # pyright: ignore[reportArgumentType]


def _stripped(**fields: object) -> Callable[[], LeafHarness]:
    class Harness(ScriptedHarness):
        def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
            return Strips(super()._build(capabilities, script), **fields)

    return Harness


def test_l26_passes_a_leaf_with_no_request_id_or_output_count_with_a_note() -> None:
    usage = Usage(input_tokens=12, output_tokens=None, raw_usage={"input_tokens": 12})
    l26 = _check(leaf_checks(_stripped(request_id=None, usage=usage)), "L26")
    assert l26 == Verdict(
        "L26",
        "pass",
        "the wire responses carry no output token count and no request id, right only for a "
        "leaf whose wire or runtime gives none",
    )


def test_l26_fails_a_leaf_with_no_input_count() -> None:
    usage = Usage(input_tokens=None, output_tokens=0, raw_usage={"output_tokens": 0})
    l26 = _check(leaf_checks(_stripped(usage=usage)), "L26")
    assert l26.result == "fail"
    assert l26.reason is not None
    assert "usage input tokens" in l26.reason


def test_l26_fails_an_empty_request_id() -> None:
    l26 = _check(leaf_checks(_stripped(request_id="")), "L26")
    assert l26.result == "fail"
    assert l26.reason is not None
    assert "request id is empty" in l26.reason


def test_p8_passes_an_outcome_with_no_request_id_with_a_note() -> None:
    class NoIds(ScriptedProviders):
        def _wrap(self, leaf: Scripted) -> Leaf:
            return Strips(leaf, request_id=None)

    p8 = _check(provider_checks(lambda: NoIds("leaf", [CAPABILITIES], CAPABILITIES)), "P8")
    assert p8 == Verdict(
        "P8",
        "pass",
        "an outcome carries no request id, right only for a leaf whose wire or runtime gives none",
    )
