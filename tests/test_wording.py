"""A question's wording: a ``noul``'s ``yes`` and ``no`` descriptions and a structured
instruction, built, measured, encoded, and sent or refused by every leaf, never dropped."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

from idia import (
    Deadline,
    Dispatcher,
    DoesNotFit,
    Fallback,
    Label,
    NoulAnswer,
    ProviderError,
    Question,
    Router,
    WireRequest,
)
from idia.core import JsonValue
from idia.fit import _question_bytes
from idia.providers import _canon, cloudflare, openai_decisions, system_one
from idia.providers.chat import openai as chat
from idia.providers.recorded import Recorded
from idia.providers.scripted import Scripted
from tests import _openai_cases as chat_cases
from tests import test_openai_chat as chat_tests
from tests import test_openai_decisions as decisions_tests
from tests import test_system_one as system_one_tests
from tests._cloudflare_harness import ClairicHarness
from tests._support import capabilities, request, text_state

YES = "The sender asks for money back."
NO = "The sender only asks a question."
PARTS: JsonValue = {"question": "Is `message` a complaint?", "marker": "STRUCTURED-PART"}


def described(yes: str | None = YES, no: str | None = NO, key: str = "q") -> Question:
    return Question(
        key=key,
        kind="noul",
        instruction="Is this a complaint?",
        labels=(Label(name="yes", description=yes), Label(name="no", description=no)),
    )


def structured(kind: str = "noul", key: str = "q") -> Question:
    if kind == "extraction":
        return Question(key=key, kind="extraction", instruction=PARTS, value_type="text")
    return Question(key=key, kind="noul", instruction=PARTS)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------------------------


def test_a_noul_takes_yes_then_no_with_either_or_both_described() -> None:
    assert described().labels[0].description == YES
    assert described(no=None).labels[1].description is None
    assert described(yes=None).labels[0].description is None
    assert Question(key="q", kind="noul", instruction="?").labels == ()


@pytest.mark.parametrize(
    "labels",
    [
        (Label(name="no", description="n"), Label(name="yes", description="y")),
        (Label(name="yes", description="y"),),
        (Label(name="true", description="y"), Label(name="false", description="n")),
        (
            Label(name="yes", description="y"),
            Label(name="no", description="n"),
            Label(name="maybe"),
        ),
    ],
    ids=["reversed", "one", "true-false", "three"],
)
def test_any_other_noul_labels_are_refused(labels: tuple[Label, ...]) -> None:
    with pytest.raises(ValueError, match="yes then no"):
        Question(key="q", kind="noul", instruction="?", labels=labels)


def test_a_noul_with_neither_described_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        described(yes=None, no=None)


@pytest.mark.parametrize("instruction", [PARTS, ("first part", {"second": "part"})])
def test_an_instruction_may_be_a_json_object_or_array(instruction: JsonValue) -> None:
    question = Question(key="q", kind="noul", instruction=instruction)
    assert question.instruction == instruction
    hash(question)  # a mapping instruction is left out of the hash


def test_an_object_instruction_is_held_read_only() -> None:
    given = {"question": "Is it so?"}
    question = Question(key="q", kind="noul", instruction=given)
    given["question"] = "changed"
    assert question.instruction == {"question": "Is it so?"}
    with pytest.raises(TypeError):
        question.instruction["question"] = "changed"  # type: ignore[index]  # pyright: ignore[reportIndexIssue]


@pytest.mark.parametrize(
    ("instruction", "error"),
    [
        (None, TypeError),
        (3, TypeError),
        (True, TypeError),
        (["a list"], TypeError),
        ({1: "x"}, TypeError),
        ({}, ValueError),
        ((), ValueError),
        ("", ValueError),
    ],
    ids=["none", "int", "bool", "list", "int-key", "empty-object", "empty-array", "empty-str"],
)
def test_any_other_instruction_is_refused(instruction: object, error: type[Exception]) -> None:
    with pytest.raises(error, match=r"Question\.instruction"):
        Question(key="q", kind="noul", instruction=instruction)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


# ----------------------------------------------------------------------------------------
# Encoding and measuring
# ----------------------------------------------------------------------------------------


def test_a_string_instruction_keeps_its_canonical_encoding() -> None:
    encoded = _canon.encode(Question(key="q", kind="noul", instruction="Is it so?"))
    assert encoded == {
        "type": "Question",
        "key": "q",
        "kind": "noul",
        "instruction": "Is it so?",
        "labels": (),
        "value_type": None,
        "is_list": False,
    }


@pytest.mark.parametrize("question", [described(), structured()], ids=["described", "structured"])
def test_the_wording_round_trips_through_the_canonical_form(question: Question) -> None:
    wire_request = WireRequest(state=text_state(), questions=(question,))
    encoded = json.loads(json.dumps(_canon.encode(wire_request)))
    decoded = _canon.decode(encoded, WireRequest)
    assert decoded == wire_request
    assert list(decoded.questions[0].instruction) == list(cast(Any, question.instruction))


def test_a_structured_instructions_key_order_is_in_the_canonical_hash() -> None:
    """Two instructions that differ only in key order are two prompts, so two recordings."""

    def hashed(instruction: JsonValue) -> str:
        question = Question(key="q", kind="noul", instruction=instruction)
        wire_request = WireRequest(state=text_state(), questions=(question,))
        return _canon.canonical_hash("leaf", "model", wire_request)

    assert hashed({"a": "1", "b": "2"}) != hashed({"b": "2", "a": "1"})


def test_fit_measures_a_structured_instruction_as_its_compact_json() -> None:
    compact = json.dumps(PARTS, separators=(",", ":"), ensure_ascii=False)
    assert _question_bytes(structured()) == len("q") + len(compact.encode())


def test_a_described_noul_is_not_held_to_max_labels() -> None:
    leaf = Scripted(capabilities=capabilities(max_labels=2, max_total_options=2), script=[])
    plan = Dispatcher(leaf).plan(request(described()))
    assert len(plan.wire_requests) == 1


def test_fit_measures_a_read_only_mapping_inside_an_instruction() -> None:
    inner = MappingProxyType({"part": "text"})
    question = Question(key="q", kind="noul", instruction={"outer": inner})
    assert _question_bytes(question) == len("q") + len('{"outer":{"part":"text"}}')


def test_fit_measures_a_nouls_descriptions() -> None:
    assert _question_bytes(described()) == len(
        "q" + "Is this a complaint?" + "yes" + YES + "no" + NO
    )


# ----------------------------------------------------------------------------------------
# Sent or refused, never dropped
# ----------------------------------------------------------------------------------------

Build = Callable[[WireRequest, Path], object]


def _scripted(wire_request: WireRequest, _: Path) -> object:
    leaf = Scripted(capabilities=capabilities(kinds=frozenset({"noul", "extraction"})), script=[{}])
    return asyncio.run(leaf.call(wire_request, Deadline.after(5))).prompt_sent


def _recorded(wire_request: WireRequest, log: Path) -> object:
    inner = Scripted(
        capabilities=capabilities(kinds=frozenset({"noul", "extraction"})), script=[{}]
    )
    path = log / "log.json"
    asyncio.run(Recorded(inner, path, "new").call(wire_request, Deadline.after(5)))
    replay = Recorded(None, path, "none", capabilities=inner.capabilities)
    return asyncio.run(replay.call(wire_request, Deadline.after(5))).prompt_sent


def _chat(wire_request: WireRequest, _: Path) -> object:
    [question] = wire_request.questions
    return chat._messages(wire_request.state, question, chat._letters(question))


BUILDS: dict[str, Build] = {
    "system-one": lambda w, _: system_one._body("jev-1.13.0", w, None)[0],
    "cloudflare-clef": lambda w, _: cloudflare._clef_body("clef", w, None)[0],
    "cloudflare-clairic": lambda w, _: cloudflare._clairic_body(w, None, None)[0],
    "openai-decisions": lambda w, _: openai_decisions._body("gpt-6-luna", w, None)[0],
    "openai-chat": _chat,
    "scripted": _scripted,
    "recorded": _recorded,
}
"""Each leaf's body as built before it sends, or, for the doubles, the prompt sent."""

SENDS: dict[str, tuple[bool, bool]] = {
    "system-one": (True, True),
    "cloudflare-clef": (True, True),
    "cloudflare-clairic": (False, False),
    "openai-decisions": (False, False),
    "openai-chat": (True, False),
    "scripted": (True, True),
    "recorded": (True, True),
}
"""Whether each leaf sends a described noul and a structured instruction; Clairic refuses a
noul as a kind it does not answer."""


@pytest.mark.parametrize("leaf", sorted(BUILDS))
@pytest.mark.parametrize("part", ["noul descriptions", "structured instruction"])
def test_every_leaf_sends_or_refuses_the_wording(leaf: str, part: str, tmp_path: Path) -> None:
    if part == "noul descriptions":
        question, texts, sends = described(), (YES, NO), SENDS[leaf][0]
    else:
        kind = "extraction" if leaf == "cloudflare-clairic" else "noul"
        question, texts, sends = structured(kind), ("STRUCTURED-PART",), SENDS[leaf][1]
    wire_request = WireRequest(state=text_state(), questions=(question,))
    if not sends:
        with pytest.raises(ProviderError) as raised:
            BUILDS[leaf](wire_request, tmp_path)
        assert not raised.value.is_retryable
        return
    built = json.dumps(BUILDS[leaf](wire_request, tmp_path), default=dict, ensure_ascii=False)
    for text in texts:
        assert text in built, (leaf, part)


@pytest.mark.parametrize("build", [system_one._body, cloudflare._clef_body])
def test_a_noul_with_descriptions_sends_criteria(build: Callable[..., Any]) -> None:
    def criteria(question: Question) -> object:
        wire_request = WireRequest(state=text_state(), questions=(question,))
        body = build("jev-1.13.0", wire_request, None)[0]
        return body["questions"]["q"].get("criteria", "absent")

    assert criteria(described()) == {"true": YES, "false": NO}
    assert list(criteria(described())) == ["true", "false"]  # type: ignore[call-overload]
    assert criteria(described(no=None)) == {"true": YES}
    assert criteria(described(yes=None)) == {"false": NO}
    assert criteria(Question(key="q", kind="noul", instruction="?")) == "absent"


@pytest.mark.parametrize("build", [system_one._body, cloudflare._clef_body])
def test_a_structured_instruction_is_sent_as_it_is(build: Callable[..., Any]) -> None:
    wire_request = WireRequest(state=text_state(), questions=(structured(),))
    body = build("jev-1.13.0", wire_request, None)[0]
    assert body["questions"]["q"]["instructions"] == PARTS


def test_system_one_sends_a_described_noul_through_its_call() -> None:
    wire = system_one_tests._answering({"q": {"type": "noul", "noul": 0.7}})
    wire_request = WireRequest(state=text_state(), questions=(described(),))
    response = system_one_tests._call(wire.leaf(), wire_request)
    assert response.answers == {"q": NoulAnswer(probability=0.7)}
    assert wire.body_sent["questions"]["q"]["criteria"] == {"true": YES, "false": NO}


@pytest.mark.parametrize("question", [described(), structured()], ids=["described", "structured"])
def test_the_decisions_leaf_refuses_the_wording_before_sending(question: Question) -> None:
    wire = decisions_tests._answering([])
    wire_request = WireRequest(state=text_state(), questions=(question,))
    error = decisions_tests._raised(wire.leaf(), wire_request)
    assert not error.is_retryable
    assert "'q'" in error.message
    assert wire.sent == []


def test_the_chat_leaf_puts_a_nouls_descriptions_on_its_yes_and_no_lines() -> None:
    question = described(no=None)
    _, user = chat._messages(text_state(), question, chat._letters(question))
    content = user["content"]
    assert isinstance(content, str)
    assert content.endswith(f"A. Yes: {YES}\nB. No")


def test_the_chat_leaf_refuses_a_structured_instruction_before_sending() -> None:
    wire = chat_tests._answering({"A": 1.0})
    with pytest.raises(ProviderError, match="structured instruction") as raised:
        chat_tests._call(wire.leaf(), structured())
    assert not raised.value.is_retryable
    assert wire.sent == []


def _record_without(field: str) -> Any:
    return capabilities(kinds=frozenset({"noul", "extraction"}), **{field: False})


@pytest.mark.parametrize(
    ("field", "questions"),
    [
        ("carries_noul_descriptions", (described(key="a"), described(key="b"))),
        ("carries_structured_instructions", (structured(key="a"), structured(key="b"))),
    ],
)
def test_a_wording_the_record_does_not_carry_does_not_fit(
    field: str, questions: tuple[Question, ...]
) -> None:
    leaf = Scripted(capabilities=_record_without(field), script=[])
    plain = Question(key="c", kind="noul", instruction="Is it so?")
    with pytest.raises(DoesNotFit) as raised:
        Dispatcher(leaf).plan(request(plain, *questions))
    error = raised.value
    assert (error.check, error.observed, error.limit) == (field, ("a", "b"), False)
    assert str(error).startswith("questions 'a', 'b' have ")
    assert str(error).endswith(f": {field} is False")
    assert leaf.calls == []


def test_a_plain_wording_fits_a_record_that_carries_neither() -> None:
    record = capabilities(carries_noul_descriptions=False, carries_structured_instructions=False)
    leaf = Scripted(capabilities=record, script=[])
    plain = Question(key="c", kind="noul", instruction="Is it so?")
    assert len(Dispatcher(leaf).plan(request(plain)).wire_requests) == 1


def _member(**fields: Any) -> Dispatcher:
    return Dispatcher(Scripted(capabilities=capabilities(**fields), script=[]))


def test_a_router_carries_noul_descriptions_only_where_its_noul_members_do() -> None:
    nouls = _member(kinds=frozenset({"noul"}), carries_structured_instructions=False)
    extracts = _member(kinds=frozenset({"extraction"}), carries_noul_descriptions=False)
    record = Router({"noul": nouls, "extraction": extracts}).capabilities
    assert record.carries_noul_descriptions is True
    assert record.carries_structured_instructions is False
    without = Router({"extraction": extracts}).capabilities
    assert without.carries_noul_descriptions is False


def test_a_fallback_carries_a_wording_only_when_every_member_does() -> None:
    both = _member()
    neither = _member(carries_noul_descriptions=False, carries_structured_instructions=False)
    record = Fallback([both, neither]).capabilities
    assert (record.carries_noul_descriptions, record.carries_structured_instructions) == (
        False,
        False,
    )
    assert Fallback([both, both]).capabilities.carries_structured_instructions is True


def test_the_chat_leaf_refuses_a_record_that_claims_structured_instructions() -> None:
    record = dataclasses.replace(chat_cases.chat_record(), carries_structured_instructions=True)
    with pytest.raises(ValueError, match="carries_structured_instructions must be false"):
        chat_tests._answering({"A": 1.0}).leaf(capabilities=record)


def test_clairic_refuses_a_structured_instruction_before_sending() -> None:
    harness = ClairicHarness()
    leaf = harness.make([])
    wire_request = WireRequest(state=text_state(), questions=(structured("extraction"),))
    with pytest.raises(ProviderError, match="structured instruction") as raised:
        asyncio.run(leaf.call(wire_request, Deadline.after(5)))
    assert not raised.value.is_retryable
    assert harness.requests_sent() == 0


def test_a_router_sends_a_structured_instruction_to_a_member_that_carries_it() -> None:
    """The router's derived flag is false, as one member's is, but its own check reads only
    kinds and modalities, so the member that answers decides."""
    nouls = Scripted(capabilities=capabilities(kinds=frozenset({"noul"})), script=[{}])
    extracts = Scripted(
        capabilities=capabilities(
            kinds=frozenset({"extraction"}), carries_structured_instructions=False
        ),
        script=[],
    )
    router = Router({"noul": Dispatcher(nouls), "extraction": Dispatcher(extracts)})
    assert router.capabilities.carries_structured_instructions is False
    response = asyncio.run(router.answer(request(structured())))
    assert len(nouls.calls) == 1
    assert [outcome.keys for outcome in response.outcomes] == [("q",)]


def test_a_parsed_instruction_is_held_read_only_all_the_way_down() -> None:
    parsed = json.loads('{"question": "Is it so?", "parts": ["a", {"b": ["c"]}]}')
    question = Question(key="q", kind="noul", instruction=parsed)
    held = cast(Any, question.instruction)
    assert held == {"question": "Is it so?", "parts": ("a", {"b": ("c",)})}
    assert isinstance(held["parts"], tuple)
    with pytest.raises(TypeError):
        held["parts"][1]["b"] = "changed"
    parsed["parts"][1]["b"].append("d")
    assert held["parts"][1]["b"] == ("c",)
