"""The shapes, the closed sets, the errors and the two protocols."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import ast
import dataclasses
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import pytest

import idia
from idia import (
    CANDIDATE_ELICITATIONS,
    CANDIDATE_KINDS,
    UNBOUNDED,
    Box,
    BoxAnswer,
    Capabilities,
    Deadline,
    DeadlineExceeded,
    ElicitationNotInRelease,
    ExtractionAnswer,
    FailedAttempt,
    Fault,
    ImagePart,
    Kind,
    KindNotInRelease,
    Label,
    LabelAnswer,
    Leaf,
    NoulAnswer,
    Outcome,
    Provider,
    ProviderError,
    Question,
    Refusal,
    Request,
    RequestFailed,
    Response,
    Rounding,
    Rule,
    RuleFailure,
    State,
    TextPart,
    TokenEstimate,
    TokenEstimator,
    Usage,
    WireRequest,
    WireResponse,
    core,
)
from idia.core import _without_keys  # pyright: ignore[reportPrivateUsage]
from tests._support import FixedEstimator, capabilities, choice, noul, png_header, text_state

# ----------------------------------------------------------------------------------------
# Capabilities
# ----------------------------------------------------------------------------------------


def test_capabilities_cannot_be_mutated() -> None:
    record = capabilities()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.max_questions = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.confidence["noul"] = "x"  # type: ignore[index]
    assert isinstance(record.confidence, MappingProxyType)
    held = {"long-context": {"choice"}}
    record = capabilities(feature_kinds=held, feature_modalities={"long-context": {"image"}})
    held["long-context"].add("box")
    with pytest.raises(TypeError):
        record.feature_kinds["images"] = frozenset()  # type: ignore[index]
    assert isinstance(record.feature_kinds, MappingProxyType)
    assert isinstance(record.feature_modalities, MappingProxyType)
    assert record.feature_kinds == {"long-context": frozenset({"choice"})}
    assert record.feature_modalities == {"long-context": frozenset({"image"})}


def test_capabilities_has_no_default_on_any_limit_field() -> None:
    limits = {
        "max_input_tokens",
        "max_question_tokens",
        "max_body_bytes",
        "max_questions",
        "max_images",
        "max_images_per_question",
        "max_labels",
        "max_levels",
        "max_total_options",
    }
    for field in dataclasses.fields(Capabilities):
        if field.name in limits:
            assert field.default is dataclasses.MISSING
            assert field.default_factory is dataclasses.MISSING
    assert all(field.kw_only for field in dataclasses.fields(Capabilities))
    assert "noul" in capabilities().kinds
    assert capabilities(max_questions=UNBOUNDED).max_questions is UNBOUNDED


def test_silent_truncation_with_unknown_token_limit_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="max_input_tokens"):
        capabilities(truncates_silently=True, max_input_tokens=None)
    assert capabilities(truncates_silently=True, max_input_tokens=24_000).truncates_silently
    assert capabilities(truncates_silently=None, max_input_tokens=None).max_input_tokens is None


def test_composite_capabilities_skip_the_truncation_refusal() -> None:
    record = capabilities(truncates_silently=True, max_input_tokens=None, composite=True)
    assert record.composite
    assert record.max_input_tokens is None


_TAKES_IMAGES: Final[dict[str, Any]] = {
    "modalities": frozenset({"text", "image"}),
    "image_carriers": frozenset({"parts"}),
    "max_images": UNBOUNDED,
    "max_images_per_question": UNBOUNDED,
}


def test_empty_image_media_types_with_the_image_modality_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="image_media_types is empty beside the image modality"):
        capabilities(**_TAKES_IMAGES, image_media_types=frozenset())
    assert capabilities(image_media_types=frozenset()).image_media_types == frozenset()
    assert capabilities(**_TAKES_IMAGES, image_media_types=None).image_media_types is None
    # A composite's derived set may be empty: its own check never reads it.
    derived = capabilities(**_TAKES_IMAGES, image_media_types=frozenset(), composite=True)
    assert derived.image_media_types == frozenset()


def test_a_zero_max_images_per_question_with_the_image_modality_is_rejected_at_construction() -> (
    None
):
    with pytest.raises(ValueError, match="beside the image modality") as raised:
        capabilities(**{**_TAKES_IMAGES, "max_images_per_question": 0})
    assert str(raised.value) == (
        "max_images_per_question is 0 beside the image modality; a cap of 0 refuses every image"
    )
    assert capabilities(max_images_per_question=0).max_images_per_question == 0
    derived = capabilities(**{**_TAKES_IMAGES, "max_images_per_question": 0}, composite=True)
    assert derived.max_images_per_question == 0


def test_a_zero_max_images_with_the_image_modality_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="beside the image modality") as raised:
        capabilities(**{**_TAKES_IMAGES, "max_images": 0})
    assert str(raised.value) == (
        "max_images is 0 beside the image modality; a cap of 0 refuses every image"
    )
    assert capabilities(max_images=0).max_images == 0
    derived = capabilities(**{**_TAKES_IMAGES, "max_images": 0}, composite=True)
    assert derived.max_images == 0


@pytest.mark.parametrize("name", ["max_image_bytes", "max_total_image_bytes"])
def test_a_zero_image_byte_cap_with_the_image_modality_is_rejected_at_construction(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="beside the image modality") as raised:
        capabilities(**_TAKES_IMAGES, image_media_types=None, **{name: 0})
    assert str(raised.value) == (
        f"{name} is 0 beside the image modality; a cap of 0 refuses every image"
    )
    assert getattr(capabilities(**{name: 0}), name) == 0
    derived = capabilities(**_TAKES_IMAGES, image_media_types=None, composite=True, **{name: 0})
    assert getattr(derived, name) == 0


@pytest.mark.parametrize(
    ("declared", "canonical"),
    [("image/PNG", "image/png"), ("image/jpg", "image/jpeg"), ("IMAGE/JPG", "image/jpeg")],
)
def test_image_media_types_must_be_lower_case_canonical(declared: str, canonical: str) -> None:
    with pytest.raises(ValueError, match="image_media_types holds") as raised:
        capabilities(**_TAKES_IMAGES, image_media_types={"image/webp", declared})
    assert str(raised.value) == (
        f"image_media_types holds {declared!r}; write it as {canonical!r}, the form an image's "
        "type is read in"
    )


def test_image_fields_are_checked_and_frozen_at_construction() -> None:
    record = capabilities(**_TAKES_IMAGES, image_media_types={"image/png"})
    assert record.image_media_types == frozenset({"image/png"})
    assert isinstance(record.image_media_types, frozenset)
    with pytest.raises(TypeError, match="image_media_types"):
        capabilities(**_TAKES_IMAGES, image_media_types=["image/png"])
    with pytest.raises(TypeError, match=r"image_media_types\[\]"):
        capabilities(**_TAKES_IMAGES, image_media_types={1})
    for name in ("max_image_bytes", "max_total_image_bytes"):
        with pytest.raises(ValueError, match=name):
            capabilities(**{name: -1})
    for bad in (-1, True):
        with pytest.raises(ValueError, match="min_image_side"):
            capabilities(min_image_side=bad)


def test_enabled_features_must_be_offered() -> None:
    with pytest.raises(ValueError, match="opt_in_features"):
        capabilities(enabled_features=frozenset({"images"}))


@pytest.mark.parametrize("field", ["feature_kinds", "feature_modalities"])
def test_feature_maps_with_a_key_outside_opt_in_features_are_rejected_at_construction(
    field: str,
) -> None:
    value = frozenset({"box"}) if field == "feature_kinds" else frozenset({"image"})
    with pytest.raises(ValueError, match=f"{field} names 'images'.*not in opt_in_features"):
        capabilities(**{field: {"images": value}})
    assert getattr(capabilities(**{field: {"long-context": value}}), field) == {
        "long-context": value
    }


@pytest.mark.parametrize("field", ["feature_kinds", "feature_modalities"])
def test_feature_map_that_is_not_a_mapping_is_rejected_at_construction(field: str) -> None:
    with pytest.raises(TypeError, match=rf"^Capabilities\.{field} must be a mapping, got list$"):
        capabilities(**{field: [("long-context", frozenset({"box"}))]})


@pytest.mark.parametrize("field", ["feature_kinds", "feature_modalities"])
def test_feature_map_value_that_is_not_a_set_is_rejected_at_construction(field: str) -> None:
    message = rf"^Capabilities\.{field}\['long-context'\] must be a set, got str$"
    with pytest.raises(TypeError, match=message):
        capabilities(**{field: {"long-context": "box"}})
    with pytest.raises(TypeError, match="must be a set, got tuple"):
        capabilities(**{field: {"long-context": ("box",)}})


@pytest.mark.parametrize("field", ["feature_kinds", "feature_modalities"])
def test_feature_map_with_an_empty_set_is_rejected_at_construction(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} maps 'long-context' to the empty set"):
        capabilities(**{field: {"long-context": frozenset()}})
    assert getattr(capabilities(**{field: {}}), field) == {}


def test_enabled_feature_whose_kinds_are_not_declared_is_rejected_at_construction() -> None:
    on = frozenset({"long-context"})
    with pytest.raises(ValueError, match="enabled feature 'long-context' adds box, which kinds"):
        capabilities(enabled_features=on, feature_kinds={"long-context": frozenset({"box"})})
    with pytest.raises(ValueError, match="adds image, which modalities does not declare"):
        capabilities(enabled_features=on, feature_modalities={"long-context": {"image"}})
    # On and declared: the map stays the same whatever is enabled.
    declared = capabilities(
        kinds=frozenset({"noul", "box"}),
        modalities=frozenset({"text", "image"}),
        max_images=UNBOUNDED,
        max_images_per_question=UNBOUNDED,
        enabled_features=on,
        feature_kinds={"long-context": frozenset({"box"})},
        feature_modalities={"long-context": frozenset({"image"})},
    )
    assert declared.feature_kinds == {"long-context": {"box"}}
    # Off: the value may be missing from the record, or already in it.
    assert capabilities(feature_kinds={"long-context": frozenset({"box"})}).feature_kinds
    assert capabilities(feature_kinds={"long-context": frozenset({"noul"})}).feature_kinds


@pytest.mark.parametrize("bad", [True, False, -1])
def test_a_limit_is_neither_a_bool_nor_negative(bad: object) -> None:
    with pytest.raises(ValueError, match="max_labels"):
        capabilities(max_labels=bad)
    assert capabilities(max_labels=0).max_labels == 0
    with pytest.raises(ValueError, match="min_levels"):
        capabilities(min_levels=bad)
    assert capabilities(min_levels=0).min_levels == 0


def test_token_estimate_refuses_a_negative_figure_or_margin() -> None:
    assert TokenEstimate(tokens=0, margin=0.0).margin == 0.0
    with pytest.raises(ValueError, match="at least 0"):
        TokenEstimate(tokens=-1, margin=0.1)
    with pytest.raises(ValueError, match="at least 0"):
        TokenEstimate(tokens=10, margin=-0.1)


@pytest.mark.parametrize("margin", [math.nan, math.inf])
def test_token_estimate_refuses_a_margin_that_is_not_finite(margin: float) -> None:
    with pytest.raises(ValueError, match=r"TokenEstimate\.margin must be finite"):
        TokenEstimate(tokens=1, margin=margin)


@pytest.mark.parametrize(
    ("tokens", "margin", "field"),
    [(True, 0.1, "tokens"), (1.5, 0.1, "tokens"), ("1", 0.1, "tokens"), (1, True, "margin")],
)
def test_token_estimate_refuses_a_field_of_the_wrong_type(
    tokens: object, margin: object, field: str
) -> None:
    with pytest.raises(TypeError, match=rf"TokenEstimate\.{field}"):
        TokenEstimate(tokens=cast(int, tokens), margin=cast(float, margin))


@pytest.mark.parametrize("bad", [1.5, math.nan, -1.5, "8"])
@pytest.mark.parametrize("field", ["max_questions", "min_levels", "min_image_side"])
def test_a_limit_that_is_not_an_int_is_refused(field: str, bad: object) -> None:
    with pytest.raises(TypeError, match=rf"Capabilities\.{field}"):
        capabilities(**{field: bad})


def test_an_int_too_large_for_a_float_is_refused_with_value_error() -> None:
    with pytest.raises(ValueError, match=r"Deadline\.at"):
        Deadline(at=10**400)
    with pytest.raises(ValueError, match=r"Deadline\.after"):
        Deadline.after(10**400)
    with pytest.raises(ValueError, match=r"Deadline\.after"):
        Deadline.after(-(10**400))
    with pytest.raises(ValueError, match="noul_complement"):
        idia.noul_complement(10**400)


# ----------------------------------------------------------------------------------------
# Questions and the kind set
# ----------------------------------------------------------------------------------------


def test_kind_set_is_closed_at_five_with_three_candidates() -> None:
    assert frozenset({"noul", "choice", "score", "extraction", "box"}) == core._KINDS
    assert frozenset({"estimate", "forecast", "multi_label"}) == CANDIDATE_KINDS
    assert frozenset({"structured_output", "verbalised", "sampled_n"}) == CANDIDATE_ELICITATIONS


def test_question_of_a_candidate_kind_raises_before_any_request() -> None:
    with pytest.raises(KindNotInRelease) as raised:
        Question(key="q", kind=cast(Kind, "estimate"), instruction="How many?")
    assert raised.value.kind == "estimate"
    assert raised.value.candidate is True

    with pytest.raises(KindNotInRelease) as raised:
        Question(
            key="q",
            kind="extraction",
            instruction="Forecast?",
            value_type=cast(core.ValueType, "forecast"),
        )
    assert raised.value.kind == "forecast"
    assert raised.value.candidate is True


def test_unknown_kind_raises_kind_not_in_release_with_candidate_false() -> None:
    with pytest.raises(KindNotInRelease) as raised:
        Question(key="q", kind=cast(Kind, "riddle"), instruction="?")
    assert raised.value.kind == "riddle"
    assert raised.value.candidate is False
    assert "riddle" in str(raised.value)


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "choice"},
        {"kind": "score"},
        {"kind": "choice", "labels": (Label(name="only"),)},
        {"kind": "choice", "labels": (Label(name="a"), Label(name="a"))},
        {"kind": "extraction"},
        {"kind": "extraction", "value_type": "colour"},
        {"kind": "noul", "labels": (Label(name="yes"), Label(name="no"))},
        {"kind": "box", "labels": (Label(name="here"),)},
        {"kind": "noul", "is_list": True},
        {"kind": "noul", "value_type": "text"},
    ],
)
def test_malformed_question_raises_value_error(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"question|label|value type|list"):
        Question(key="q", instruction="?", **fields)  # type: ignore[arg-type]


def test_well_formed_questions_build() -> None:
    assert noul().labels == ()
    assert choice().labels[0].description == "the colour red"
    listed = Question(
        key="q", kind="extraction", instruction="Names?", value_type="text", is_list=True
    )
    assert listed.is_list
    assert Question(key="q", kind="score", instruction="?", labels=(Label(name="one"),)).kind


def test_request_keys_are_unique() -> None:
    deadline = Deadline.after(1)
    with pytest.raises(ValueError, match="unique"):
        Request(state=text_state(), questions=(noul("a"), noul("a")), deadline=deadline)
    request = Request(state=text_state(), questions=(noul("a"), noul("b")), deadline=deadline)
    assert [question.key for question in request.questions] == ["a", "b"]
    wire = WireRequest(state=request.state, questions=request.questions)
    assert wire.questions is request.questions


def test_request_refuses_zero_questions() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        Request(state=text_state(), questions=(), deadline=Deadline.after(1))


def test_image_part_holds_bytes() -> None:
    part = ImagePart(data=b"\x89PNG", media_type="image/png")
    assert part.data == b"\x89PNG"
    assert ImagePart(data=b"\x89PNG").media_type is None
    assert not hasattr(part, "url")
    assert core.JsonPart(value={"a": (1, 2.5, None, True)}).value == {"a": (1, 2.5, None, True)}


def _refused(**fields: Any) -> Callable[[], object]:
    """A builder of the shape named by ``fields["of"]`` with the other fields given."""
    shape = fields.pop("of")
    return lambda: shape(**fields)


_QUESTION = {"key": "q", "kind": "noul", "instruction": "?"}
_LABELLED = {"key": "q", "kind": "choice", "instruction": "?"}
_REQUEST = {"state": State(parts=()), "questions": (Question(**_QUESTION),)}  # type: ignore[arg-type]

REFUSED = [
    (_refused(of=TextPart, text=b"x"), TypeError, "TextPart.text must be a str, got bytes"),
    (_refused(of=ImagePart, data="x"), TypeError, "ImagePart.data must be bytes, got str"),
    (
        _refused(of=ImagePart, data=bytearray(b"x")),
        TypeError,
        "ImagePart.data must be bytes, got bytearray",
    ),
    (
        _refused(of=ImagePart, data=b"x", media_type=3),
        TypeError,
        "ImagePart.media_type must be a str, got int",
    ),
    (
        _refused(of=ImagePart, data=b"x", media_type=""),
        ValueError,
        "ImagePart.media_type must not be empty; use None when unknown",
    ),
    (_refused(of=State, parts=[]), TypeError, "State.parts must be a tuple, got list"),
    (
        _refused(of=State, parts=(TextPart(text="x"), "y")),
        TypeError,
        "State.parts[1] must be a TextPart, JsonPart or ImagePart, got str",
    ),
    (_refused(of=Label, name=None), TypeError, "Label.name must be a str, got NoneType"),
    (_refused(of=Label, name=""), ValueError, "Label.name must not be empty"),
    (
        _refused(of=Label, name="a", description=1),
        TypeError,
        "Label.description must be a str, got int",
    ),
    (
        _refused(of=Label, name="a", description=""),
        ValueError,
        "Label.description must not be empty; use None for no description",
    ),
    (
        _refused(of=Question, **{**_QUESTION, "key": 1}),
        TypeError,
        "Question.key must be a str, got int",
    ),
    (
        _refused(of=Question, **{**_QUESTION, "key": ""}),
        ValueError,
        "Question.key must not be empty",
    ),
    (
        _refused(of=Question, **{**_QUESTION, "kind": ["noul"]}),
        TypeError,
        "Question.kind must be a str, got list",
    ),
    (
        _refused(of=Question, **{**_QUESTION, "instruction": None}),
        TypeError,
        "Question.instruction must be a str, a JSON object or a JSON array (a tuple), not NoneType",
    ),
    (
        _refused(of=Question, **{**_QUESTION, "instruction": ""}),
        ValueError,
        "Question.instruction must not be empty",
    ),
    (
        _refused(of=Question, **_LABELLED, labels=[Label(name="a"), Label(name="b")]),
        TypeError,
        "Question.labels must be a tuple, got list",
    ),
    (
        _refused(of=Question, **_LABELLED, labels=(Label(name="a"), "b")),
        TypeError,
        "Question.labels[1] must be a Label, got str",
    ),
    (
        _refused(of=Question, key="q", kind="extraction", instruction="?", value_type=["text"]),
        TypeError,
        "Question.value_type must be a str, got list",
    ),
    (
        _refused(of=Question, **{**_QUESTION, "is_list": 0}),
        TypeError,
        "Question.is_list must be a bool, got int",
    ),
    (
        _refused(of=Request, **{**_REQUEST, "state": "x"}, deadline=Deadline.after(1)),
        TypeError,
        "Request.state must be a State, got str",
    ),
    (
        _refused(of=Request, **{**_REQUEST, "questions": []}, deadline=Deadline.after(1)),
        TypeError,
        "Request.questions must be a tuple, got list",
    ),
    (
        _refused(of=Request, **{**_REQUEST, "questions": ("q",)}, deadline=Deadline.after(1)),
        TypeError,
        "Request.questions[0] must be a Question, got str",
    ),
    (
        _refused(of=Request, **_REQUEST, deadline=30.0),
        TypeError,
        "Request.deadline must be a Deadline, got float",
    ),
    (
        _refused(of=Request, **_REQUEST, deadline=Deadline.after(1), options=[("a", 1)]),
        TypeError,
        "Request.options must be a mapping, got list",
    ),
    (
        _refused(of=WireRequest, **{**_REQUEST, "state": None}),
        TypeError,
        "WireRequest.state must be a State, got NoneType",
    ),
    (
        _refused(of=WireRequest, **{**_REQUEST, "questions": [Question(**_QUESTION)]}),  # type: ignore[arg-type]
        TypeError,
        "WireRequest.questions must be a tuple, got list",
    ),
    (
        _refused(of=WireRequest, **{**_REQUEST, "questions": (None,)}),
        TypeError,
        "WireRequest.questions[0] must be a Question, got NoneType",
    ),
    (
        _refused(of=WireRequest, **_REQUEST, options=None),
        TypeError,
        "WireRequest.options must be a mapping, got NoneType",
    ),
    (_refused(of=Deadline, at="5"), TypeError, "Deadline.at must be a number, got str"),
    (_refused(of=Deadline, at=True), TypeError, "Deadline.at must be a number, got bool"),
    (_refused(of=Deadline, at=math.nan), ValueError, "Deadline.at must be finite, got nan"),
    (_refused(of=Deadline, at=math.inf), ValueError, "Deadline.at must be finite, got inf"),
]


@pytest.mark.parametrize(("build", "error", "message"), REFUSED)
def test_a_constructor_refuses_a_field_naming_it(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error) as raised:
        build()
    assert type(raised.value) is error
    assert str(raised.value) == message


def test_a_field_is_type_checked_before_it_is_checked_for_emptiness_or_by_the_kind_rules() -> None:
    # An empty list is the wrong type before it is too few labels or questions.
    with pytest.raises(TypeError, match=r"Question\.labels must be a tuple"):
        Question(key="q", kind="choice", instruction="?", labels=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"Request\.questions must be a tuple"):
        Request(state=State(parts=()), questions=[], deadline=Deadline.after(1))  # type: ignore[arg-type]
    # A kind that is not a str is a TypeError, not KindNotInRelease.
    with pytest.raises(TypeError, match=r"Question\.kind must be a str"):
        Question(key="q", kind=None, instruction="?")  # type: ignore[arg-type]


def test_what_the_constructors_still_take() -> None:
    assert TextPart(text="").text == ""
    assert State(parts=()).parts == ()
    assert core.JsonPart(value=cast(Any, object())).value is not None  # leaves and Fit check it
    assert Label(name="a", description=None).description is None
    options = MappingProxyType({"a": 1})
    request = Request(**_REQUEST, deadline=Deadline.after(1), options=options)  # type: ignore[arg-type]
    assert dict(request.options) == {"a": 1}
    assert Deadline(at=3).at == 3


@pytest.mark.parametrize("endless", [-math.inf, math.inf, math.nan], ids=["-inf", "inf", "nan"])
def test_deadline_after_refuses_seconds_that_are_not_finite_with_its_own_message(
    endless: float,
) -> None:
    with pytest.raises(ValueError, match=r"^Deadline\.after takes a finite") as raised:
        Deadline.after(endless)
    assert str(raised.value) == (
        "Deadline.after takes a finite number of seconds, at most 30 days (2592000 seconds), "
        f"got {endless!r}"
    )


@pytest.mark.parametrize("seconds", [0, 0.0, -5.0], ids=["zero", "zero-float", "negative"])
def test_deadline_after_allows_a_past_instant(seconds: float) -> None:
    assert Deadline.after(seconds).passed()


def test_a_deadline_more_than_thirty_days_ahead_is_refused_as_a_wall_clock_time() -> None:
    with pytest.raises(ValueError, match="wall-clock") as raised:
        Deadline(at=time.time() + 30)
    assert str(raised.value) == (
        "Deadline.at is more than 30 days ahead on the monotonic clock; a wall-clock time was "
        "likely given: use Deadline.after(seconds)"
    )
    with pytest.raises(ValueError, match=r"^Deadline\.after takes") as after:
        Deadline.after(core._DEADLINE_HORIZON + 60)
    assert str(after.value) == (
        "Deadline.after takes a finite number of seconds, at most 30 days (2592000 seconds), "
        "got 2592060.0"
    )
    assert Deadline.after(core._DEADLINE_HORIZON - 60).remaining() > 0
    assert Deadline(at=-1e12).remaining() < 0  # a past instant stays legal


# ----------------------------------------------------------------------------------------
# Deadline
# ----------------------------------------------------------------------------------------


def test_deadline_after_counts_down_on_the_monotonic_clock() -> None:
    deadline = Deadline.after(10)
    assert 0 < deadline.remaining() <= 10
    assert Deadline(at=0.0).remaining() < 0


@pytest.mark.parametrize("passed", [True, False])
def test_a_deadline_within_one_clock_tick_has_passed(
    monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    monkeypatch.setattr(core, "_CLOCK_TICK", 60.0 if passed else -60.0)
    assert Deadline.after(5).passed() is passed
    assert Deadline(at=-1e12).passed()


def test_a_deadline_within_one_float_ulp_of_the_clock_has_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 2.0**50  # the clock's ulp here is 0.25 seconds
    monkeypatch.setattr(core, "_CLOCK_TICK", -60.0)
    monkeypatch.setattr(core.time, "monotonic", lambda: now)
    assert Deadline(at=now + 0.25).passed()
    assert not Deadline(at=now + 0.5).passed()


# ----------------------------------------------------------------------------------------
# Elicitations
# ----------------------------------------------------------------------------------------


def test_elicitation_not_in_release_names_the_candidate() -> None:
    with pytest.raises(ElicitationNotInRelease) as raised:
        core.check_elicitation("verbalised")
    assert raised.value.elicitation == "verbalised"
    assert raised.value.candidate is True

    with pytest.raises(ElicitationNotInRelease) as raised:
        core.check_elicitation("tarot")
    assert raised.value.candidate is False

    assert idia.check_elicitation is core.check_elicitation
    assert core.check_elicitation("logprobs") == "logprobs"
    assert core.check_elicitation("passthrough") == "passthrough"


# ----------------------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------------------


def test_provider_error_carries_the_retry_facts_and_nothing_else() -> None:
    error = ProviderError("rate limited", is_retryable=True, retry_after=2.5, status_code=429)
    assert error.is_retryable
    assert error.retry_after == 2.5
    assert error.status_code == 429
    assert error.request_id is None
    assert error.message == "rate limited"
    assert not hasattr(error, "body")
    assert ProviderError("bad request").is_retryable is False


def test_deadline_exceeded_is_a_non_retryable_provider_error() -> None:
    error = DeadlineExceeded()
    assert isinstance(error, ProviderError)
    assert error.is_retryable is False
    assert "deadline" in str(error)


def _outcome(error: ProviderError | None, failed: int = 0) -> Outcome:
    attempts = tuple(
        FailedAttempt(
            provider="fake",
            model_requested="fake-1",
            error=ProviderError("boom", is_retryable=True),
            request_id=None,
            started_at=0.0,
            duration=0.1,
        )
        for _ in range(failed)
    )
    return Outcome(
        rounding=Rounding(probability_decimals=2, score_decimals=2),
        keys=("q",),
        error=error,
        provider="fake",
        model_requested="fake-1",
        model_reported=None,
        snapshot=None,
        elicitation=None if error else "passthrough",
        calibrated=None if error else True,
        prompt_sent={"body": "..."} if error is None else None,
        usage=None,
        latency=0.2,
        started_at=0.0,
        request_id=None,
        failed_attempts=attempts,
        warnings=(),
    )


def test_attempts_are_derived_from_failed_attempts() -> None:
    assert _outcome(error=None).attempts == 1
    assert _outcome(error=None, failed=2).attempts == 3
    assert _outcome(error=DeadlineExceeded()).attempts == 0
    assert _outcome(error=ProviderError("x"), failed=3).attempts == 3


def test_response_has_no_provenance_and_outcome_error_is_a_provider_error_or_none() -> None:
    answered = _outcome(error=None)
    failed = _outcome(error=ProviderError("x"))
    response = Response(answers={"q": NoulAnswer(probability=0.5)}, outcomes=(answered, failed))
    assert not hasattr(response, "provenance")
    assert response.outcomes[0].error is None
    assert isinstance(response.outcomes[1].error, ProviderError)
    assert isinstance(response.answers, MappingProxyType)
    assert isinstance(answered.prompt_sent, MappingProxyType)
    raised = RequestFailed(response)
    assert raised.response is response
    assert str(raised) == "no wire request answered: fake fake-1: x (1 outcome failed)"


def test_request_failed_names_the_first_failed_outcome_and_counts_the_failures() -> None:
    first = _outcome(error=ProviderError("HTTP 401: bad key"))
    others = (_outcome(error=ProviderError("HTTP 503")), _outcome(error=DeadlineExceeded()))
    raised = RequestFailed(Response(answers={}, outcomes=(first, *others)))
    assert str(raised) == (
        "no wire request answered: fake fake-1: HTTP 401: bad key "
        "(3 outcomes failed; the first shown)"
    )


def test_request_failed_falls_back_to_the_error_class_name_for_an_empty_message() -> None:
    raised = RequestFailed(Response(answers={}, outcomes=(_outcome(error=ProviderError("")),)))
    assert str(raised) == "no wire request answered: fake fake-1: ProviderError (1 outcome failed)"


def test_request_failed_with_no_failed_outcome_gives_the_count_alone() -> None:
    def message(count: int) -> str:
        outcomes = tuple(_outcome(error=None) for _ in range(count))
        return str(RequestFailed(Response(answers={}, outcomes=outcomes)))

    assert message(0) == "no wire request answered (0 outcomes)"
    assert message(1) == "no wire request answered (1 outcome)"
    assert message(2) == "no wire request answered (2 outcomes)"


def test_wire_response_and_usage_maps_are_read_only() -> None:
    usage = Usage(input_tokens=10, output_tokens=2, raw_usage={"total": 12})
    wire = WireResponse(
        answers={"q": Refusal()},
        elicitation="passthrough",
        calibrated=True,
        prompt_sent={"questions": ()},
        usage=usage,
        raw_body=b"{}",
    )
    assert isinstance(wire.answers, MappingProxyType)
    assert isinstance(wire.prompt_sent, MappingProxyType)
    assert isinstance(usage.raw_usage, MappingProxyType)
    assert (
        WireResponse(
            answers={}, elicitation="logprobs", calibrated=False, prompt_sent="A?"
        ).prompt_sent
        == "A?"
    )
    assert Usage(input_tokens=None, output_tokens=None).raw_usage == {}
    assert not hasattr(wire, "provider")


# ----------------------------------------------------------------------------------------
# Answer shapes
# ----------------------------------------------------------------------------------------


def test_every_published_shape_is_a_frozen_slotted_dataclass() -> None:
    shapes = [
        core.State,
        core.TextPart,
        core.JsonPart,
        core.ImagePart,
        core.Label,
        Question,
        Request,
        WireRequest,
        WireResponse,
        Outcome,
        Response,
        core.Notice,
        Capabilities,
        core.Rounding,
        Deadline,
        Usage,
        FailedAttempt,
        TokenEstimate,
        LabelAnswer,
        NoulAnswer,
        ExtractionAnswer,
        BoxAnswer,
        Box,
        core.Span,
        Refusal,
        Fault,
        RuleFailure,
    ]
    for shape in shapes:
        params: Any = getattr(shape, "__dataclass_params__")  # noqa: B009
        assert params.frozen, shape
        assert "__slots__" in vars(shape), shape
        assert all(field.kw_only for field in dataclasses.fields(shape)), shape


def test_label_answer_echo_names_what_it_echoes() -> None:
    with pytest.raises(ValueError, match="echo_of"):
        LabelAnswer(probabilities=(("a", 1.0),), echo=("a",))
    answer = LabelAnswer(probabilities=(("a", 1.0),), echo=("a",), echo_of="names")
    assert answer.off_label_mass is None
    assert answer.derived == frozenset()
    assert NoulAnswer(probability=0.3).derived == frozenset()
    assert ExtractionAnswer(value=None, p_null=1.0).guess is None
    assert BoxAnswer(region=Box(top=0, left=0, bottom=1, right=1)).confidence is None
    assert Rule.candidate_shape.value == "candidate_shape"
    assert len(Rule) == 18


# ----------------------------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------------------------


class _MinimalLeaf:
    def __init__(self) -> None:
        self.name = "minimal"
        self.model = "minimal-1"
        self.capabilities = capabilities()
        self.estimator = FixedEstimator()

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        return WireResponse(answers={}, elicitation="passthrough", calibrated=True)


@dataclasses.dataclass(frozen=True)
class _FrozenLeaf:
    name: str
    model: str
    capabilities: Capabilities
    estimator: TokenEstimator

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        return WireResponse(answers={}, elicitation="passthrough", calibrated=True)


class _ForwardingProvider:
    def __init__(self, leaf: Leaf) -> None:
        self._leaf = leaf

    @property
    def capabilities(self) -> Capabilities:
        return self._leaf.capabilities

    async def answer(self, request: Request) -> Response:
        return Response(answers={}, outcomes=())


def test_leaf_and_provider_are_runtime_checkable_and_a_leaf_is_not_a_provider() -> None:
    leaf = _MinimalLeaf()
    assert isinstance(leaf, Leaf)
    assert not isinstance(leaf, Provider)
    assert isinstance(leaf.estimator, TokenEstimator)
    assert leaf.estimator.estimate(text_state(), [noul()]).tokens == 10
    assert not isinstance(object(), Leaf)


def test_a_frozen_dataclass_leaf_and_a_forwarding_provider_satisfy_the_protocols() -> None:
    # The annotations are the check: pyright accepts a frozen field and a property where
    # the protocol declares a read-only property.
    frozen: Leaf = _FrozenLeaf(
        name="frozen", model="frozen-1", capabilities=capabilities(), estimator=FixedEstimator()
    )
    provider: Provider = _ForwardingProvider(frozen)
    assert isinstance(frozen, Leaf)
    assert not isinstance(frozen, Provider)
    assert isinstance(provider, Provider)
    assert provider.capabilities is frozen.capabilities


# ----------------------------------------------------------------------------------------
# The package
# ----------------------------------------------------------------------------------------


def test_core_imports_only_the_standard_library() -> None:
    source = Path(core.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imported.add(node.module.partition(".")[0])
    assert imported, "core.py imports something"
    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


def test_public_api_is_exactly_what_all_lists() -> None:
    public = {
        name
        for name, value in vars(idia).items()
        if not name.startswith("_") and not isinstance(value, type(sys))
    }
    assert public == set(idia.__all__)
    assert "provider" in public
    assert "ProviderNotFound" in public
    assert "DoesNotFit" in public
    assert "FitPlan" in public
    assert "noul_complement" in public


# ----------------------------------------------------------------------------------------
# Request options
# ----------------------------------------------------------------------------------------


def test_request_options_are_empty_by_default_and_read_only() -> None:
    state = idia.State(parts=(idia.TextPart(text="x"),))
    question = idia.Question(key="q", kind="noul", instruction="?")
    request = idia.Request(state=state, questions=(question,), deadline=idia.Deadline.after(1.0))
    assert dict(request.options) == {}
    held = {"reference_date": "2026-10-21"}
    optioned = dataclasses.replace(request, options=held)
    held["reference_date"] = "changed"
    assert optioned.options == {"reference_date": "2026-10-21"}
    with pytest.raises(TypeError):
        optioned.options["more"] = 1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        optioned.options = {}  # type: ignore[misc]
    wire = idia.WireRequest(state=state, questions=(question,), options={"a": 1})
    assert isinstance(wire.options, MappingProxyType)
    assert dict(idia.WireRequest(state=state, questions=(question,)).options) == {}
    assert hash(wire) == hash(dataclasses.replace(wire, options={"a": 2}))
    assert wire != dataclasses.replace(wire, options={"a": 2})


def _items_all_the_way_down(value: object) -> object:
    """``value`` with every mapping as its items list, so a comparison sees key order."""
    if isinstance(value, Mapping):
        held = cast(Mapping[str, object], value)
        return [(key, _items_all_the_way_down(item)) for key, item in held.items()]
    if isinstance(value, (list, tuple)):
        return [_items_all_the_way_down(item) for item in cast(Sequence[object], value)]
    return value


def test_a_json_part_is_held_read_only_in_the_order_given() -> None:
    built: dict[str, Any] = {"z": 1, "a": {"y": [{"q": 1, "b": 2}], "c": None}, "m": "x"}
    part = core.JsonPart(value=built)
    built["z"] = 99
    built["a"]["y"][0]["q"] = 99
    built["a"]["y"].append(3)
    built["new"] = True
    assert _items_all_the_way_down(part.value) == [
        ("z", 1),
        ("a", [("y", [[("q", 1), ("b", 2)]]), ("c", None)]),
        ("m", "x"),
    ]
    value = cast(Mapping[str, Any], part.value)
    assert isinstance(value, MappingProxyType)
    assert isinstance(value["a"], MappingProxyType)
    assert isinstance(value["a"]["y"], tuple)
    assert isinstance(value["a"]["y"][0], MappingProxyType)
    with pytest.raises(TypeError):
        value["z"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        value["a"]["y"][0]["q"] = 2  # type: ignore[index]
    listed = core.JsonPart(value=cast(Any, [{"b": 1, "a": 2}, [1, 2]]))
    assert _items_all_the_way_down(listed.value) == [[("b", 1), ("a", 2)], [1, 2]]
    assert isinstance(listed.value, tuple)
    assert core.JsonPart(value="text").value == "text"


def test_a_structured_instruction_is_held_read_only_in_the_order_given() -> None:
    built: dict[str, Any] = {"z": "ask", "a": {"y": [{"q": 1, "b": 2}], "c": None}}
    question = Question(key="q", kind="noul", instruction=built)
    built["z"] = "changed"
    built["a"]["y"][0]["q"] = 99
    built["a"]["y"].append(3)
    built["new"] = True
    assert _items_all_the_way_down(question.instruction) == [
        ("z", "ask"),
        ("a", [("y", [[("q", 1), ("b", 2)]]), ("c", None)]),
    ]
    held = cast(Mapping[str, Any], question.instruction)
    assert isinstance(held, MappingProxyType)
    assert isinstance(held["a"], MappingProxyType)
    assert isinstance(held["a"]["y"], tuple)
    assert isinstance(held["a"]["y"][0], MappingProxyType)
    with pytest.raises(TypeError):
        held["z"] = "again"  # type: ignore[index]
    with pytest.raises(TypeError):
        held["a"]["y"][0]["q"] = 2  # type: ignore[index]


def test_a_value_that_contains_itself_or_nests_too_deep_is_refused_at_build() -> None:
    loop: list[Any] = [1]
    loop.append({"again": loop})
    with pytest.raises(ValueError, match=r"^JsonPart\.value contains itself$"):
        core.JsonPart(value=cast(Any, loop))
    looped: dict[str, Any] = {"a": 1}
    looped["self"] = (looped,)
    with pytest.raises(ValueError, match=r"^Question\.instruction contains itself$"):
        Question(key="q", kind="noul", instruction=looped)
    state, asked = State(parts=()), (Question(key="q", kind="noul", instruction="?"),)
    options = cast(Any, {"k": loop})
    with pytest.raises(ValueError, match=r"^Request\.options\['k'\] contains itself$"):
        Request(state=state, questions=asked, deadline=Deadline.after(1), options=options)
    with pytest.raises(ValueError, match=r"^WireRequest\.options\['k'\] contains itself$"):
        idia.WireRequest(state=state, questions=asked, options=options)
    deep: Any = 1
    for _ in range(100_000):
        deep = [{"k": deep}]
    with pytest.raises(ValueError, match=r"^JsonPart\.value nests too deep$"):
        core.JsonPart(value=deep)
    with pytest.raises(ValueError, match=r"^Question\.instruction nests too deep$"):
        Question(key="q", kind="noul", instruction={"k": deep})
    shared = {"b": 1}
    twice = cast(tuple[Any, ...], core.JsonPart(value=cast(Any, [shared, shared])).value)
    assert all(isinstance(item, MappingProxyType) for item in twice)  # shared, not a cycle


def test_options_are_held_read_only_in_the_order_given() -> None:
    state = idia.State(parts=(idia.TextPart(text="x"),))
    question = idia.Question(key="q", kind="noul", instruction="?")
    built: dict[str, Any] = {"z": {"y": 1, "b": [{"d": 1, "c": 2}]}, "a": [1, 2]}
    request = idia.Request(
        state=state, questions=(question,), deadline=idia.Deadline.after(1.0), options=built
    )
    wire = idia.WireRequest(state=state, questions=(question,), options=built)
    built["z"]["y"] = 99
    built["z"]["b"][0]["d"] = 99
    built["a"].append(3)
    expected = [("z", [("y", 1), ("b", [[("d", 1), ("c", 2)]])]), ("a", [1, 2])]
    for options in (request.options, wire.options):
        assert _items_all_the_way_down(options) == expected
        inner = cast(Mapping[str, Any], options["z"])
        assert isinstance(inner, MappingProxyType)
        assert isinstance(inner["b"], tuple)
        assert isinstance(options["a"], tuple)
        with pytest.raises(TypeError):
            inner["y"] = 2  # type: ignore[index]


def test_a_state_with_a_json_object_hashes() -> None:
    built: Any = {"a": [1, {"b": 2}], "c": 3}
    state = State(parts=(core.JsonPart(value=built), TextPart(text="x")))
    reordered = State(parts=(core.JsonPart(value={"c": 3, "a": built["a"]}), TextPart(text="x")))
    assert hash(state) == hash(reordered)
    assert state == reordered  # equal shapes, two prompts: == and hash() ignore key order
    one, other = core.JsonPart(value={"b": 1, "a": 2}), core.JsonPart(value={"a": 2, "b": 1})
    assert hash(one) == hash(other)
    assert one == other
    request = Request(state=state, questions=(Question(**_QUESTION),), deadline=Deadline.after(1))  # type: ignore[arg-type]
    assert isinstance(hash(request), int)


def test_unknown_options_names_every_unknown_key_in_one_warning() -> None:
    assert idia.unknown_options({}, frozenset()) == ()
    assert idia.unknown_options({"a": 1}, frozenset({"a"})) == ()
    [warning] = idia.unknown_options({"b": 1, "a": 2, "c": 3}, frozenset({"c"}))
    assert warning.code == idia.WARNING_UNKNOWN_OPTION == "unknown-option"
    assert warning.field == "options"
    assert warning.message.endswith("a, b")


# ----------------------------------------------------------------------------------------
# Reading an image
# ----------------------------------------------------------------------------------------


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload


_SOF = b"\x08" + (290).to_bytes(2, "big") + (300).to_bytes(2, "big") + b"\x03"
_JFIF = _segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")


@pytest.mark.parametrize(
    ("data", "size"),
    [
        pytest.param(png_header(300, 290), (300, 290), id="png"),
        pytest.param(b"\xff\xd8" + _JFIF + _segment(0xC0, _SOF), (300, 290), id="jpeg-baseline"),
        pytest.param(
            b"\xff\xd8" + _segment(0xC4, b"\x00" * 4) + b"\xff\xff\xd0" + _segment(0xC2, _SOF),
            (300, 290),
            id="jpeg-progressive-after-tables-fill-and-restart",
        ),
        pytest.param(b"GIF89a" + b"\x00" * 20, None, id="other-format"),
        pytest.param(png_header()[:20], None, id="png-cut-short"),
        pytest.param(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, None, id="png-without-ihdr"),
        pytest.param(b"\xff\xd8" + _segment(0xDA, b"\x00" * 8), None, id="jpeg-scan-first"),
        pytest.param(b"\xff\xd8\xff\xd9\x00\x00", None, id="jpeg-end-first"),
        pytest.param(b"\xff\xd8" + _JFIF + b"\x00\x00\x00\x00", None, id="jpeg-no-marker"),
        pytest.param(b"\xff\xd8" + _segment(0xC0, _SOF)[:8], None, id="jpeg-frame-cut-short"),
        pytest.param(b"\xff\xd8" + _JFIF, None, id="jpeg-no-frame"),
    ],
)
def test_the_image_size_is_read_from_the_header(data: bytes, size: tuple[int, int] | None) -> None:
    assert core._image_size(data) == size


_PNG_BYTES: Final = png_header()
_JPEG_BYTES: Final = b"\xff\xd8\xff\xe0"
_WEBP_BYTES: Final = b"RIFF\x00\x00\x00\x00WEBPVP8 "
_PNG_JPEG: Final = frozenset({"image/png", "image/jpeg"})


@pytest.mark.parametrize(
    ("data", "declared", "accepted", "read"),
    [
        pytest.param(_PNG_BYTES, "image/png", _PNG_JPEG, "image/png", id="declared-in-the-set"),
        pytest.param(_JPEG_BYTES, "IMAGE/JPG", _PNG_JPEG, "image/jpeg", id="declared-canonical"),
        pytest.param(_PNG_BYTES, "image/gif", _PNG_JPEG, "image/png", id="falls-to-signature"),
        pytest.param(_WEBP_BYTES, "image/gif", _PNG_JPEG, None, id="neither-in-the-set"),
        pytest.param(_WEBP_BYTES, None, frozenset({"image/webp"}), "image/webp", id="webp"),
        pytest.param(b"\x00\x01", None, _PNG_JPEG, None, id="no-signature"),
        pytest.param(_PNG_BYTES, "image/gif", None, "image/gif", id="unknown-set-declared"),
        pytest.param(_JPEG_BYTES, None, None, "image/jpeg", id="unknown-set-signature"),
        pytest.param(b"\x00\x01", None, None, None, id="unknown-set-nothing"),
    ],
)
def test_an_image_media_type_is_read_as_the_record_judges_it(
    data: bytes, declared: str | None, accepted: frozenset[str] | None, read: str | None
) -> None:
    assert core._image_media_type(ImagePart(data=data, media_type=declared), accepted) == read


# ----------------------------------------------------------------------------------------
# A noul's complement
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yes", "no"),
    [
        pytest.param(0.1, 0.9, id="one-decimal"),
        pytest.param(0.7, 0.3, id="the-float-noise-case"),
        pytest.param(0.73, 0.27, id="two-decimals"),
        pytest.param(0.9876, 0.0124, id="four-decimals"),
        pytest.param(0.123456789012345, 0.876543210987655, id="full-precision"),
        pytest.param(5e-324, 1.0, id="the-smallest-float"),
        pytest.param(0, 1.0, id="zero"),
        pytest.param(1, 0.0, id="one"),
    ],
)
def test_noul_complement_is_the_exact_decimal_complement(yes: float, no: float) -> None:
    complement = idia.noul_complement(yes)
    assert complement == no
    assert type(complement) is float
    with localcontext() as context:
        context.prec = 400
        assert complement == float(Decimal(1) - Decimal(repr(float(yes))))


def test_noul_complement_carries_none_of_the_float_subtractions_noise() -> None:
    assert 1 - 0.7 == 0.30000000000000004
    assert idia.noul_complement(0.7) == 0.3


def test_noul_complement_reads_a_float_subclass_by_its_value() -> None:
    class Probability(float):
        def __repr__(self) -> str:
            return f"Probability({float(self)!r})"

    assert idia.noul_complement(Probability(0.7)) == 0.3


@pytest.mark.parametrize("outside", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_noul_complement_refuses_a_probability_outside_the_unit_interval(outside: float) -> None:
    with pytest.raises(ValueError, match="takes a probability from 0 to 1") as raised:
        idia.noul_complement(outside)
    assert str(raised.value) == f"noul_complement takes a probability from 0 to 1, got {outside!r}"


@pytest.mark.parametrize("wrong", [True, "0.5", None])
def test_noul_complement_refuses_what_is_not_a_number(wrong: object) -> None:
    with pytest.raises(TypeError, match="noul_complement takes a number"):
        idia.noul_complement(cast(float, wrong))


def test_noul_complement_leaves_the_answer_unchanged() -> None:
    answer = NoulAnswer(probability=0.735)
    before = answer.probability
    assert idia.noul_complement(answer.probability) == 0.265
    assert answer.probability is before
    assert answer == NoulAnswer(probability=0.735)


# ----------------------------------------------------------------------------------------
# The credential filter on a provider error's text
# ----------------------------------------------------------------------------------------

_KEY: Final = "sk-proj-abcdefghijklmnopqrstuvwxyz012345"
_TOKEN: Final = "Bearer abcdefghijklmnop0123456789"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(f"HTTP 401: invalid key {_KEY}", id="sk-key"),
        pytest.param(f"the wire echoed Authorization: {_TOKEN}", id="bearer"),
        pytest.param(f"Illegal header value b'{_TOKEN}\\n'", id="bearer-in-a-header-error"),
    ],
)
def test_a_provider_error_text_never_carries_a_key_shaped_substring(text: str) -> None:
    error = ProviderError(text, status_code=401, request_id="req_76f189c39eda412dbdf7edef202ea06b")
    for shown in (error.message, str(error), *error.args):
        assert _KEY not in shown
        assert "abcdefghijklmnop0123456789" not in shown
        assert "<filtered>" in shown
    assert error.request_id == "req_76f189c39eda412dbdf7edef202ea06b"
    assert ProviderError(error.message).message == error.message  # filtering again is a no-op


def test_a_request_failed_message_carries_no_key_shaped_substring() -> None:
    error = ProviderError(f"HTTP 401: invalid key {_KEY}", status_code=401)
    outcome = Outcome(
        rounding=Rounding(probability_decimals=2, score_decimals=2),
        keys=("q",),
        error=error,
        provider="p",
        model_requested="m",
        model_reported=None,
        snapshot=None,
        elicitation=None,
        calibrated=None,
        prompt_sent=None,
        usage=None,
        latency=0.0,
        started_at=0.0,
        request_id=None,
        failed_attempts=(),
        warnings=(),
    )
    message = str(RequestFailed(Response(answers={}, outcomes=(outcome,))))
    assert _KEY not in message
    assert "no wire request answered: p m: HTTP 401: invalid key <filtered>" in message


def test_text_that_is_not_key_shaped_passes_unchanged() -> None:
    for text in (
        "Incorrect API key provided: sk-idia-***alid.",
        "task-sk-abcdefghijklmnopqrstuvwx",
        "the request id is req_76f189c39eda412dbdf7edef202ea06b",
    ):
        assert ProviderError(text).message == text


# A credential of no key shape, as a Cloudflare token or a TypeSafe key is.
_CREDENTIAL: Final = "cf-token-0123456789abcdefABCDEF"


def test_a_given_credential_is_filtered_out_of_text_wherever_it_appears() -> None:
    text = f"invalid token {_CREDENTIAL}; again: {_CREDENTIAL}."
    assert _without_keys(text, _CREDENTIAL) == "invalid token <filtered>; again: <filtered>."
    assert _without_keys(text) == text  # not key-shaped, so nothing given leaves it as it is


def test_a_credential_holding_another_is_filtered_whole() -> None:
    inner = _CREDENTIAL[:12]
    assert _without_keys(f"key {_CREDENTIAL}", inner, _CREDENTIAL) == "key <filtered>"


@pytest.mark.parametrize("credential", ["", "k", "1234567"])
def test_a_credential_shorter_than_the_floor_is_left_in_the_text(credential: str) -> None:
    text = "HTTP 401: invalid credentials 1234567"
    assert _without_keys(text, credential) == text


def test_a_credential_at_the_floor_is_filtered() -> None:
    assert _without_keys("key 12345678", "12345678") == "key <filtered>"
