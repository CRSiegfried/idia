"""The harness that runs the leaf half against the scripted leaf.

Each exchange becomes a script item, answered by question key from the harness's cases, so
one table serves every wire request a check or a provider sends. What a script cannot
honestly stand in for, a body to read, a status mapper, an SDK or an HTTP client, is not
applicable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Final

from idia.conformance._base import (
    CandidateShape,
    CheckNotApplicable,
    Error,
    Exchange,
    Failure,
    Perturbed,
    RefusalOnWire,
    Stall,
    Unreadable,
)
from idia.core import (
    UNBOUNDED,
    Answer,
    Box,
    BoxAnswer,
    Capabilities,
    Deadline,
    ExtractionAnswer,
    Kind,
    Label,
    LabelAnswer,
    Leaf,
    MappedAnswer,
    NoulAnswer,
    ProviderError,
    Question,
    Refusal,
    Rounding,
    State,
    TextPart,
    WireRequest,
)
from idia.providers.scripted import Item, Reply, Scripted, ScriptExhausted, ScriptFunction

_COLOURS: Final = (Label(name="red"), Label(name="green"), Label(name="blue"))
_LEVELS: Final = (Label(name="low"), Label(name="middle"), Label(name="high"))

CASES: Final[Mapping[Kind, tuple[tuple[Question, MappedAnswer], ...]]] = {
    "noul": (
        (
            Question(key="noul_a", kind="noul", instruction="Is it so?"),
            NoulAnswer(probability=0.25),
        ),
        (
            Question(key="noul_b", kind="noul", instruction="Is it not?"),
            NoulAnswer(probability=0.8),
        ),
    ),
    "choice": (
        (
            Question(key="choice_a", kind="choice", instruction="Which?", labels=_COLOURS),
            LabelAnswer(
                probabilities=(("red", 0.5), ("green", 0.3), ("blue", 0.2)),
                confidence=0.5,
                chosen="red",
            ),
        ),
        (
            Question(key="choice_b", kind="choice", instruction="Which else?", labels=_COLOURS),
            LabelAnswer(
                probabilities=(("red", 0.1), ("green", 0.1), ("blue", 0.8)), confidence=0.8
            ),
        ),
    ),
    "score": (
        (
            Question(key="score_a", kind="score", instruction="How much?", labels=_LEVELS),
            LabelAnswer(probabilities=(("low", 0.2), ("middle", 0.5), ("high", 0.3))),
        ),
        (
            Question(key="score_b", kind="score", instruction="How far?", labels=_LEVELS),
            LabelAnswer(probabilities=(("low", 0.6), ("middle", 0.3), ("high", 0.1))),
        ),
    ),
    "extraction": (
        (
            Question(key="extraction_a", kind="extraction", instruction="Who?", value_type="text"),
            ExtractionAnswer(value="Ada", p_null=0.1),
        ),
        (
            Question(key="extraction_b", kind="extraction", instruction="Whom?", value_type="text"),
            ExtractionAnswer(value=None, guess="Bob", p_null=0.7),
        ),
    ),
    "box": (
        (
            Question(key="box_a", kind="box", instruction="Where is the cat?"),
            BoxAnswer(region=Box(top=0.1, left=0.2, bottom=0.5, right=0.6)),
        ),
        (
            Question(key="box_b", kind="box", instruction="Where is the dog?"),
            BoxAnswer(region=Box(top=0.4, left=0.0, bottom=0.9, right=0.3)),
        ),
    ),
}
"""Two questions per kind with the answers the scripted wire gives them; every answer
passes validation under two-decimal rounding."""

STATE: Final = State(parts=(TextPart(text="The state every conformance question is about."),))

CAPABILITIES: Final = Capabilities(
    kinds=frozenset({"noul", "choice", "score", "extraction", "box"}),
    modalities=frozenset({"text"}),
    image_carriers=frozenset(),
    max_input_tokens=100_000,
    max_question_tokens=UNBOUNDED,
    max_body_bytes=UNBOUNDED,
    max_questions=UNBOUNDED,
    max_images=0,
    max_images_per_question=0,
    max_image_bytes=0,
    max_total_image_bytes=0,
    max_labels=20,
    max_levels=20,
    max_total_options=UNBOUNDED,
    min_levels=2,
    min_image_side=0,
    image_media_types=frozenset(),
    key_patterns=(),
    recounts_state_per_question=False,
    returns_refusal=False,
    calibrated=True,
    snapshot_pinnable=True,
    truncates_silently=False,
    confidence={"choice": "top-probability"},
    rounding=Rounding(probability_decimals=2, score_decimals=2),
    opt_in_features=frozenset({"long-context"}),
    enabled_features=frozenset(),
    feature_kinds={},
    feature_modalities={},
)
"""The scripted leaf's record under the suite: every kind, text only, one opt-in feature."""

_EXTRA_KEY: Final = "conformance_extra"


class ScriptedHarness:
    """A :class:`~idia.conformance.LeafHarness` over the scripted leaf.

    ``cases`` maps each kind to its questions and the answers the scripted wire gives
    them; a key not in the cases is left unanswered. A subclass overrides ``_build`` to put
    a different leaf over the same script, and ``_refuse`` and ``_item`` to make more
    exchanges.
    """

    prompt_hidden: bool = False

    def __init__(
        self,
        capabilities: Capabilities = CAPABILITIES,
        *,
        cases: Mapping[Kind, Sequence[tuple[Question, MappedAnswer]]] = CASES,
        name: str = "scripted",
        model: str = "scripted-1",
    ) -> None:
        self._capabilities = capabilities
        self._cases = {kind: tuple(pairs) for kind, pairs in cases.items()}
        self._answers: dict[str, MappedAnswer] = {
            question.key: answer for pairs in self._cases.values() for question, answer in pairs
        }
        self.name = name
        self.model = model
        self._sent = 0

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def make(self, exchanges: Sequence[Exchange], features: frozenset[str] | None = None) -> Leaf:
        for exchange in exchanges:
            self._refuse(exchange)
        self._sent = 0
        record = self.capabilities_for(
            self._capabilities.enabled_features if features is None else features
        )
        return self._build(record, self._script(iter(tuple(exchanges))))

    def capabilities_for(self, features: frozenset[str]) -> Capabilities:
        """The record with ``features`` listed as enabled: a scripted feature changes no
        other field."""
        return replace(self._capabilities, enabled_features=features)

    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> Leaf:
        """The leaf under test, answering from ``script``."""
        return Scripted(capabilities=capabilities, script=script, name=self.name, model=self.model)

    def _script(self, exchanges: Iterator[Exchange]) -> ScriptFunction:
        async def script(wire_request: WireRequest, deadline: Deadline) -> Item:
            self._sent += 1
            exchange = next(exchanges, None)
            if exchange is None:
                raise ScriptExhausted(f"the harness's exchanges ran out at call {self._sent}")
            if isinstance(exchange, Stall):
                await asyncio.sleep(max(deadline.remaining(), 0.0) + 60.0)
            return self._item(exchange, wire_request)

        return script

    def _item(self, exchange: Exchange, wire_request: WireRequest) -> Item:
        if isinstance(exchange, Failure):
            status = 400 if exchange.status is None else exchange.status
            return ProviderError(
                f"scripted failure, status {status}",
                is_retryable=status in (408, 429) or status >= 500,
                status_code=status,
            )
        if isinstance(exchange, Error):
            return ProviderError(f"scripted {exchange.which} error", is_retryable=True)
        answers = {
            question.key: self._answers[question.key]
            for question in wire_request.questions
            if question.key in self._answers
        }
        if isinstance(exchange, Perturbed):
            _perturb(answers, wire_request, exchange)
        if isinstance(exchange, RefusalOnWire):
            refused = [q.key for q in wire_request.questions if q.kind == exchange.kind]
            answers.update(dict.fromkeys(refused[:1], Refusal()))
        return Reply(answers=answers)

    def _refuse(self, exchange: Exchange) -> None:
        """Raise :class:`CheckNotApplicable` for what a script cannot honestly stand in for."""
        if isinstance(exchange, Unreadable):
            raise CheckNotApplicable("the scripted leaf reads no body")
        if isinstance(exchange, CandidateShape):
            raise CheckNotApplicable("the scripted leaf has no wire format to map from")
        if isinstance(exchange, Failure) and (
            exchange.retry_after is not None or exchange.request_id is not None
        ):
            raise CheckNotApplicable(
                "the scripted leaf has no status mapper: an error carries only what the "
                "script gave it"
            )

    def wire_request(self, kind: Kind) -> WireRequest:
        return WireRequest(state=STATE, questions=tuple(q for q, _ in self._cases[kind]))

    def expected(self, kind: Kind) -> Mapping[str, Answer]:
        return {question.key: answer for question, answer in self._cases[kind]}

    def requests_sent(self) -> int:
        return self._sent

    def sdk_retries(self) -> int:
        raise CheckNotApplicable("the scripted leaf is built on no SDK")

    async def client_closed(self, *, injected: bool) -> bool:
        raise CheckNotApplicable("the scripted leaf has no HTTP client")


def _perturb(answers: dict[str, MappedAnswer], wire_request: WireRequest, how: Perturbed) -> None:
    """Change the first answer of ``how.kind`` as ``how`` says; a wire request with no
    question of that kind is answered as recorded."""
    keys = [q.key for q in wire_request.questions if q.kind == how.kind and q.key in answers]
    if not keys:
        return
    key = keys[0]
    answer = answers[key]
    if how.how == "keys":
        answers[_EXTRA_KEY] = answers.pop(key)
    elif how.how == "noul_range" and isinstance(answer, NoulAnswer):
        answers[key] = replace(answer, probability=1.5)
    elif how.how == "sum" and isinstance(answer, LabelAnswer):
        low = min(range(len(answer.probabilities)), key=lambda i: answer.probabilities[i][1])
        moved = tuple(
            (name, p + 0.25 if i == low else p) for i, (name, p) in enumerate(answer.probabilities)
        )
        answers[key] = replace(answer, probabilities=moved)
    else:
        raise ValueError(f"cannot perturb a {how.kind} answer by {how.how!r}")
