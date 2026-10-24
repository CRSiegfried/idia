"""The OpenAI chat leaf: label probabilities read from a chat model by the ``logprobs``
elicitation, through the ``openai`` SDK's Chat Completions.

:class:`OpenAIChat` asks one question per call. The labels are coded as letters in the
order asked, A for the first, and for a ``noul`` A for yes and B for no; the prompt asks
for the letter alone. The call requests the top 20 logprobs of a one-token answer at
temperature 0 with no ``logit_bias``. The read protocol is fixed:

- read the first answer token's ``top_logprobs`` only;
- sum the probability of each letter's surface forms: the letter, with a leading space,
  in either case;
- renormalize the summed masses over the labels asked, and record the mass outside them
  as the answer's ``off_label_mass``;
- a ``noul`` is A's share of A and B; a ``choice`` derives ``chosen``, the first label of
  highest probability; a ``score`` derives ``chosen`` and ``expected_level``, each named
  in the answer's ``derived`` set.

A letter absent from the top logprobs reads as zero mass, its mass at most the smallest
probability returned, with a warning naming the label and its letter. Empty logprobs,
no label's letter in the top logprobs at all, and an off-label mass above the bound the
leaf was built with are faults, never guesses. It is registered as ``openai-chat``,
installed by the ``openai`` extra.
"""

from __future__ import annotations

import math
import re
import string
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from openai import AsyncOpenAI

from idia.answers import validate_answers
from idia.core import (
    WARNING_LABEL_ABSENT_FROM_LOGPROBS,
    WARNING_NO_SNAPSHOT,
    Capabilities,
    Deadline,
    Fault,
    JsonPart,
    JsonValue,
    LabelAnswer,
    MappedAnswer,
    Notice,
    NoulAnswer,
    ProviderError,
    Question,
    Rounding,
    Rule,
    RuleFailure,
    State,
    TextPart,
    WireRequest,
    WireResponse,
    check_elicitation,
    unknown_options,
)
from idia.providers._openai import (
    REQUEST_ID_HEADER,
    OpenAIEstimator,
    OpenAILeafBase,
    compact,
    read_body,
    unreadable,
    usage,
)
from idia.providers.http import client_versions, wire_unreported

TOP_LOGPROBS: Final = 20
"""The top-k Chat Completions allows, and so the label and level cap."""

TEMPERATURE: Final = 0.0
SYSTEM_PROMPT: Final = "Answer with the letter of one option only."

_LETTERS: Final = string.ascii_uppercase[:TOP_LOGPROBS]
_SNAPSHOT: Final = re.compile(r"-[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")

# The estimator: a chat tokenizer reads about 4 characters of English a token; the
# overhead covers the system prompt, the message framing and the letter lines' letters.
# On the recorded requests gpt-4.1 counted 58 to 70 prompt tokens, under the estimate.
_OVERHEAD_TOKENS: Final = 30
_CHARS_PER_TOKEN: Final = 4
_MARGIN: Final = 0.1


def _record() -> Capabilities:
    """The record the chat leaf declares."""
    return Capabilities(
        kinds=frozenset({"noul", "choice", "score"}),
        modalities=frozenset({"text", "json"}),
        image_carriers=frozenset(),
        max_input_tokens=None,
        max_question_tokens=None,
        max_body_bytes=None,
        max_questions=1,
        max_images=0,
        max_images_per_question=0,
        max_image_bytes=0,
        max_total_image_bytes=0,
        max_labels=TOP_LOGPROBS,
        max_levels=TOP_LOGPROBS,
        max_total_options=TOP_LOGPROBS,
        min_levels=1,
        min_image_side=0,
        image_media_types=frozenset(),
        key_patterns=(),
        recounts_state_per_question=False,
        returns_refusal=False,
        calibrated=False,
        snapshot_pinnable=True,
        carries_noul_descriptions=True,
        carries_structured_instructions=False,
        truncates_silently=None,
        confidence={},
        rounding=Rounding(probability_decimals=None, score_decimals=None),
        opt_in_features=frozenset(),
        enabled_features=frozenset(),
        feature_kinds={},
        feature_modalities={},
    )


def _check_record(record: Capabilities) -> None:
    """Refuse a caller's record that contradicts what the protocol fixes."""
    wrong = [
        what
        for what, holds in (
            ("calibrated must be false", not record.calibrated),
            (
                "carries_structured_instructions must be false",
                not record.carries_structured_instructions,
            ),
            ("the confidence map must be empty", not record.confidence),
            (
                "rounding must be none",
                record.rounding == Rounding(probability_decimals=None, score_decimals=None),
            ),
            ("max_questions must be 1", record.max_questions == 1),
            ("max_labels must be at most 20", _within_top_k(record.max_labels)),
            ("max_levels must be at most 20", _within_top_k(record.max_levels)),
            ("the modalities must not include image", "image" not in record.modalities),
            (
                "the image fields must be 0, 0, 0 and no media types",
                (
                    record.max_image_bytes,
                    record.max_total_image_bytes,
                    record.min_image_side,
                    record.image_media_types,
                )
                == (0, 0, 0, frozenset()),
            ),
        )
        if not holds
    ]
    if wrong:
        raise ValueError(f"the chat leaf's record is fixed by its protocol: {'; '.join(wrong)}")


def _within_top_k(limit: object) -> bool:
    return isinstance(limit, int) and not isinstance(limit, bool) and limit <= TOP_LOGPROBS


class OpenAIChat(OpenAILeafBase):
    """A leaf for one chat model on OpenAI's Chat Completions, read by ``logprobs``.

    ``model`` is required: there is no default chat model. ``elicitation`` must be
    ``logprobs``; a candidate elicitation, or any other name not admitted, raises
    ``ElicitationNotInRelease`` and ``passthrough``, a decision model's, ``ValueError``.
    ``max_off_label_mass`` bounds the mass outside the labels, unbounded by default.
    ``api_key`` defaults to ``OPENAI_API_KEY``; ``client`` is an injected ``AsyncOpenAI``,
    used with its retries at zero and never closed by the leaf; ``capabilities`` replaces
    the record.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        elicitation: str = "logprobs",
        max_off_label_mass: float | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        if check_elicitation(elicitation) != "logprobs":
            raise ValueError(f"the chat leaf reads by logprobs, not {elicitation!r}")
        if not model:
            raise ValueError("the chat leaf needs a model")
        if max_off_label_mass is not None and not 0.0 <= max_off_label_mass <= 1.0:
            raise ValueError("max_off_label_mass is a fraction in [0, 1]")
        if capabilities is not None:
            _check_record(capabilities)
        super().__init__(api_key=api_key, client=client, who="the chat leaf")
        self._model = model
        self._bound = max_off_label_mass
        self._capabilities = capabilities if capabilities is not None else _record()
        self._estimator = OpenAIEstimator(
            overhead=_OVERHEAD_TOKENS,
            per_question=0,
            state_chars=_CHARS_PER_TOKEN,
            question_chars=_CHARS_PER_TOKEN,
            image_tokens=0,
            margin=_MARGIN,
        )

    @property
    def name(self) -> str:
        return "openai-chat"

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def estimator(self) -> OpenAIEstimator:
        return self._estimator

    @property
    def max_off_label_mass(self) -> float | None:
        """The bound on the mass outside the labels; ``None`` is unbounded."""
        return self._bound

    def __repr__(self) -> str:
        return f"OpenAIChat(model={self._model!r})"

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Ask the one question once and return its answer, read by the protocol and
        validated, with provenance."""
        if len(wire_request.questions) != 1:
            raise ProviderError(
                f"the chat leaf asks one question per call; this wire request has "
                f"{len(wire_request.questions)}"
            )
        [question] = wire_request.questions
        letters = _letters(question)
        messages = _messages(wire_request.state, question, letters)
        response = await self._send(
            deadline,
            lambda: self._client.chat.completions.with_raw_response.create(
                model=self._model,
                messages=cast(Any, messages),
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                temperature=TEMPERATURE,
                max_completion_tokens=1,
            ),
        )
        read = read_body(response)
        choices = read.get("choices")
        if not isinstance(choices, tuple) or not choices:
            raise unreadable(response, "the body holds no choices")
        model_reported = read.get("model")
        raw_usage = read.get("usage")
        if not isinstance(model_reported, str) or not model_reported:
            raise unreadable(response, "the body fails the top-level schema")
        if not isinstance(raw_usage, Mapping):
            raise unreadable(response, "the body fails the top-level schema")
        top = _top_logprobs(choices[0])
        mapped, absent_warnings = _read(question, letters, top, self._bound)
        answers, warnings = validate_answers(
            wire_request.questions, {question.key: mapped}, self._capabilities.rounding
        )
        warnings = (*absent_warnings, *warnings)
        snapshot = model_reported if _SNAPSHOT.search(model_reported) else None
        if snapshot is None:
            warnings += (
                Notice(
                    code=WARNING_NO_SNAPSHOT,
                    message=f"the wire reported {model_reported!r}, not a dated snapshot",
                    field="snapshot",
                ),
            )
        normalized = usage(
            cast(Mapping[str, JsonValue], raw_usage), "prompt_tokens", "completion_tokens"
        )
        request_id = response.headers.get(REQUEST_ID_HEADER)
        return WireResponse(
            answers=answers,
            elicitation="logprobs",
            calibrated=self._capabilities.calibrated,
            model_reported=model_reported,
            snapshot=snapshot,
            prompt_sent={
                "messages": messages,
                "model": self._model,
                "temperature": TEMPERATURE,
                "max_completion_tokens": 1,
                "logprobs": True,
                "top_logprobs_requested": TOP_LOGPROBS,
                "top_logprobs_returned": len(top) if isinstance(top, tuple) else 0,
                "letters": dict(letters),
                "max_off_label_mass": self._bound,
            },
            usage=normalized,
            request_id=request_id,
            raw_body=read,
            warnings=warnings + unknown_options(wire_request.options, frozenset()),
            client_versions=client_versions("openai", "httpx2"),
            unreported=wire_unreported(snapshot=snapshot, usage=normalized, request_id=request_id),
        )


# ----------------------------------------------------------------------------------------
# The prompt
# ----------------------------------------------------------------------------------------


def _letters(question: Question) -> tuple[tuple[str, str], ...]:
    """Each letter and what it stands for, in the order asked: for a ``noul``, A for yes
    and B for no."""
    if question.kind == "noul":
        return (("A", "Yes"), ("B", "No"))
    if len(question.labels) > len(_LETTERS):
        raise ProviderError(
            f"the chat leaf codes at most {len(_LETTERS)} labels; {question.key!r} has "
            f"{len(question.labels)}"
        )
    return tuple(zip(_LETTERS, (label.name for label in question.labels), strict=False))


def _state_text(state: State) -> str:
    """The state as text: each part in order, a JSON part as compact JSON."""
    texts: list[str] = []
    for part in state.parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        elif isinstance(part, JsonPart):
            texts.append(compact(part.value))
        else:
            raise ProviderError("the chat leaf takes no image")
    return "\n\n".join(texts)


def _messages(
    state: State, question: Question, letters: Sequence[tuple[str, str]]
) -> tuple[dict[str, JsonValue], ...]:
    """The system message and the user message: the state, the question and one line per
    letter, ``A. <name>`` or ``A. <name>: <description>``; a described ``noul``'s ``yes``
    and ``no`` descriptions go on its ``Yes`` and ``No`` lines. A structured instruction is
    refused before sending, since the prompt is text."""
    if not isinstance(question.instruction, str):
        raise ProviderError(
            f"the chat leaf sends an instruction as text only: question {question.key!r} has "
            "a structured instruction"
        )
    descriptions: list[str | None] = [label.description for label in question.labels]
    if not descriptions:
        descriptions = [None] * len(letters)
    lines = [
        f"{letter}. {name}" if description is None else f"{letter}. {name}: {description}"
        for (letter, name), description in zip(letters, descriptions, strict=True)
    ]
    user = "\n\n".join((_state_text(state), question.instruction, "\n".join(lines)))
    return (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )


# ----------------------------------------------------------------------------------------
# The read protocol
# ----------------------------------------------------------------------------------------


class _Shape(Exception):
    """The logprobs are not the wire's shape; becomes the question's fault."""


def _top_logprobs(choice: JsonValue) -> tuple[JsonValue, ...] | str | None:
    """The first answer token's top logprobs; ``None`` where the wire returned none, a
    type's name where the shape is wrong."""
    if not isinstance(choice, Mapping):
        return type(choice).__name__
    logprobs = cast(Mapping[str, JsonValue], choice).get("logprobs")
    if logprobs is None:
        return None
    if not isinstance(logprobs, Mapping):
        return type(logprobs).__name__
    content = cast(Mapping[str, JsonValue], logprobs).get("content")
    if content is None or content == ():
        return None
    if not isinstance(content, tuple):
        return type(content).__name__
    first = content[0]
    if not isinstance(first, Mapping):
        return type(first).__name__
    top = cast(Mapping[str, JsonValue], first).get("top_logprobs")
    if top is None or top == ():
        return None
    if not isinstance(top, tuple):
        return type(top).__name__
    return top


def _logprobs(top: Sequence[JsonValue]) -> dict[str, list[float]]:
    """Each token's logprobs, several where the wire lists a token twice. A logprob that
    is not a finite number at most zero is no logprob."""
    found: dict[str, list[float]] = {}
    for entry in top:
        if not isinstance(entry, Mapping):
            raise _Shape(type(entry).__name__)
        fields = cast(Mapping[str, JsonValue], entry)
        token = fields.get("token")
        logprob = fields.get("logprob")
        if not isinstance(token, str):
            raise _Shape(type(token).__name__)
        if isinstance(logprob, bool) or not isinstance(logprob, int | float):
            raise _Shape(type(logprob).__name__)
        try:
            value = float(logprob)
        except OverflowError:
            raise _Shape("int beyond float") from None
        if not math.isfinite(value):
            raise _Shape(f"logprob {value}")
        if value > 0:
            raise _Shape(f"logprob {value} above 0")
        found.setdefault(token, []).append(value)
    return found


def _logsumexp(values: Sequence[float]) -> float:
    """``log(sum(exp(v)))``, computed so that no term underflows to a zero total."""
    highest = max(values)
    return highest + math.log(math.fsum([math.exp(value - highest) for value in values]))


def _surface_forms(letter: str) -> tuple[str, ...]:
    return (letter, " " + letter, letter.lower(), " " + letter.lower())


def _off_label_mass(label_masses: Sequence[float]) -> float:
    """The mass of the top logprobs outside the labels: one less their summed masses, at
    least zero. The protocol records it; it is no probability of an answer."""
    return max(0.0, 1.0 - math.fsum(label_masses))


def _fault(
    question: Question, failures: Sequence[RuleFailure], mapped: MappedAnswer | None
) -> Fault:
    return Fault(key=question.key, failures=tuple(failures), mapped=mapped)


def _read(
    question: Question,
    letters: Sequence[tuple[str, str]],
    top: tuple[JsonValue, ...] | str | None,
    bound: float | None,
) -> tuple[MappedAnswer, tuple[Notice, ...]]:
    """The answer the protocol reads from the first token's top logprobs, or its fault,
    with a warning where some letters were outside the top-k and read as zero mass."""
    if top is None:
        failure = RuleFailure(rule=Rule.empty_logprobs, observed=None, bound=None)
        return _fault(question, [failure], None), ()
    if isinstance(top, str):
        failure = RuleFailure(rule=Rule.kind_shape, observed=top, bound=None)
        return _fault(question, [failure], None), ()
    try:
        logprobs = _logprobs(top)
    except _Shape as shape:
        failure = RuleFailure(rule=Rule.kind_shape, observed=str(shape), bound=None)
        return _fault(question, [failure], None), ()
    ceiling = math.exp(min(min(values) for values in logprobs.values()))
    found = {
        letter: [lp for form in _surface_forms(letter) for lp in logprobs.get(form, ())]
        for letter, _ in letters
    }
    absent = [(letter, name) for letter, name in letters if not found[letter]]
    if len(absent) == len(letters):
        failures = [
            RuleFailure(rule=Rule.label_absent_from_logprobs, observed=letter, bound=None)
            for letter, _ in absent
        ]
        return _fault(question, failures, None), ()
    warnings: tuple[Notice, ...] = ()
    if absent:
        warnings = (
            Notice(
                code=WARNING_LABEL_ABSENT_FROM_LOGPROBS,
                message=f"{', '.join(f'{name} ({letter})' for letter, name in absent)} not in "
                f"the top {len(top)} logprobs, read as zero mass; each at most {ceiling:.3g}, "
                "the smallest probability returned",
                field=question.key,
            ),
        )
    # Renormalized in log space, so letters far down the top-k never sum to zero.
    present = {letter: _logsumexp(lps) for letter, lps in found.items() if lps}
    total = _logsumexp(list(present.values()))
    shares = [
        math.exp(present[letter] - total) if letter in present else 0.0 for letter, _ in letters
    ]
    off = _off_label_mass([math.exp(lp) for lp in present.values()])
    answer: LabelAnswer | NoulAnswer
    if question.kind == "noul":
        answer = NoulAnswer(probability=shares[0], off_label_mass=off)
    else:
        names = [name for _, name in letters]
        highest = max(shares)
        chosen = names[shares.index(highest)]
        if question.kind == "choice":
            answer = LabelAnswer(
                probabilities=tuple(zip(names, shares, strict=True)),
                chosen=chosen,
                off_label_mass=off,
                derived=frozenset({"chosen"}),
            )
        else:
            answer = LabelAnswer(
                probabilities=tuple(zip(names, shares, strict=True)),
                chosen=chosen,
                expected_level=math.fsum([index * share for index, share in enumerate(shares)]),
                off_label_mass=off,
                derived=frozenset({"chosen", "expected_level"}),
            )
    if bound is not None and off > bound:
        failure = RuleFailure(rule=Rule.off_label_mass_above_bound, observed=off, bound=bound)
        return _fault(question, [failure], answer), warnings
    return answer, warnings
