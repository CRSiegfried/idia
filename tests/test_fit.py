"""Fit: preflight, packing, state overflow and the merge."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Final

import pytest

from idia import (
    UNBOUNDED,
    WARNING_OVERFLOW_POLICY,
    WARNING_UNCHECKED_LIMIT,
    Deadline,
    DoesNotFit,
    Fail,
    ImagePart,
    JsonPart,
    Label,
    Notice,
    NoulAnswer,
    Outcome,
    OverflowPolicy,
    ProviderError,
    Question,
    Request,
    Rounding,
    State,
    TextPart,
    TokenEstimate,
    TokenEstimator,
)
from idia.fit import FitPlan, _check_composite, _merge, _preflight
from idia.providers.scripted import Scripted

from ._support import box, capabilities, choice, extraction, noul, png_header, score, text_state

DEADLINE = Deadline.after(60)


class Linear:
    """One token per character of text, ``per_question`` per question and ``overhead``
    once per estimate, so a chunk's estimate is not the sum of its questions'."""

    def __init__(self, per_question: int = 10, overhead: int = 5, margin: float = 0.0) -> None:
        self.per_question = per_question
        self.overhead = overhead
        self.margin = margin
        self.calls: list[tuple[State, tuple[str, ...]]] = []

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        self.calls.append((state, tuple(question.key for question in questions)))
        text = sum(len(part.text) for part in state.parts if isinstance(part, TextPart))
        tokens = text + self.per_question * len(questions) + self.overhead
        return TokenEstimate(tokens=tokens, margin=self.margin)


class Shorten:
    """A named policy that keeps the first ``keep`` characters of each text part."""

    def __init__(self, keep: int) -> None:
        self.keep = keep
        self.applied: list[int] = []

    @property
    def name(self) -> str:
        return "shorten"

    def apply(self, state: State, limit: int, estimator: TokenEstimator) -> State:
        self.applied.append(limit)
        parts = tuple(
            TextPart(text=part.text[: self.keep]) if isinstance(part, TextPart) else part
            for part in state.parts
        )
        return State(parts=parts)


def _request(*questions: Question, state: State | None = None) -> Request:
    return Request(state=state or text_state(), questions=questions, deadline=DEADLINE)


def _plan(
    request: Request,
    estimator: TokenEstimator | None = None,
    overflow: OverflowPolicy | None = None,
    **overrides: Any,
) -> FitPlan:
    overrides.setdefault("max_question_tokens", UNBOUNDED)
    return _preflight(request, capabilities(**overrides), estimator or Linear(), overflow or Fail())


def _keys(plan: FitPlan) -> list[tuple[str, ...]]:
    return [tuple(q.key for q in wire.questions) for wire in plan.wire_requests]


def _fields(warnings: tuple[Notice, ...]) -> list[str | None]:
    return [warning.field for warning in warnings]


def _outcome(*keys: str, error: ProviderError | None = None) -> Outcome:
    return Outcome(
        rounding=Rounding(probability_decimals=2, score_decimals=2),
        keys=keys,
        error=error,
        provider="scripted",
        model_requested="scripted-1",
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


def test_fail_is_the_default_overflow_policy_and_satisfies_the_protocol() -> None:
    assert isinstance(Fail(), OverflowPolicy)
    assert isinstance(Shorten(1), OverflowPolicy)
    assert Fail().name == "fail"


def test_request_that_fits_is_one_wire_request_with_the_state() -> None:
    request = _request(noul("a"), choice("b"), score("c"))
    plan = _plan(request)
    assert len(plan.wire_requests) == 1
    assert plan.wire_requests[0].state == request.state
    assert plan.wire_requests[0].questions == request.questions
    assert plan.outcome_warnings == ((),)
    assert plan.response_warnings == ()


def test_request_that_fits_whole_is_one_wire_request_even_for_a_non_monotone_estimator() -> None:
    class TwoIsExpensive(Linear):
        """Any two questions estimate far over the limit; three or one do not."""

        def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
            base = super().estimate(state, questions)
            extra = 1_000 if len(questions) == 2 else 0
            return TokenEstimate(tokens=base.tokens + extra, margin=0.0)

    estimator = TwoIsExpensive()
    request = _request(noul("a"), noul("b"), noul("c"))
    plan = _plan(request, estimator, max_input_tokens=100)
    assert _keys(plan) == [("a", "b", "c")]
    assert [keys for _, keys in estimator.calls] == [(), ("a", "b", "c")]


def test_capability_failure_raises_before_any_call() -> None:
    leaf = Scripted(capabilities=capabilities(), script=[])
    request = _request(noul("a"), extraction("b"))
    with pytest.raises(DoesNotFit) as raised:
        _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert raised.value.check == "kinds"
    assert raised.value.observed == frozenset({"extraction"})
    assert raised.value.limit == leaf.capabilities.kinds
    assert not raised.value.state_overflow
    assert str(raised.value) == "kinds not supported: extraction; no feature of this model adds it"
    assert leaf.calls == []


@pytest.mark.parametrize(
    ("request_", "overrides", "check"),
    [
        (_request(noul(), state=State(parts=(JsonPart(value={"a": 1}),))), {}, "modalities"),
        (_request(noul(), state=State(parts=(ImagePart(data=b"x"),))), {}, "modalities"),
        (
            _request(noul(), state=State(parts=(ImagePart(data=b"x"),) * 3)),
            {
                "modalities": frozenset({"text", "image"}),
                "image_carriers": frozenset({"parts"}),
                "max_images": 2,
                "max_images_per_question": 5,
            },
            "max_images",
        ),
        (
            _request(noul(), state=State(parts=(ImagePart(data=b"x"),) * 3)),
            {
                "modalities": frozenset({"text", "image"}),
                "image_carriers": frozenset({"parts"}),
                "max_images": 5,
                "max_images_per_question": 2,
            },
            "max_images_per_question",
        ),
        (_request(choice(names=("a", "b", "c"))), {"max_labels": 2}, "max_labels"),
        (_request(score(levels=4)), {"max_levels": 3}, "max_levels"),
        (_request(score(levels=2)), {"min_levels": 3}, "min_levels"),
        (_request(noul("a"), noul("b")), {"max_questions": 0}, "max_questions"),
        (_request(noul()), {"max_question_tokens": 15}, "max_question_tokens"),
    ],
)
def test_does_not_fit_check_names_the_field(
    request_: Request, overrides: dict[str, Any], check: str
) -> None:
    with pytest.raises(DoesNotFit) as raised:
        _plan(request_, **overrides)
    assert raised.value.check == check
    assert raised.value.message == str(raised.value)


def test_image_without_a_carrier_does_not_fit() -> None:
    request = _request(noul(), state=State(parts=(ImagePart(data=b"png"),)))
    with pytest.raises(DoesNotFit) as raised:
        _plan(
            request,
            modalities=frozenset({"text", "image"}),
            max_images=1,
            max_images_per_question=1,
        )
    assert raised.value.check == "image_carriers"
    assert raised.value.observed == 1
    assert raised.value.limit == frozenset()


def test_image_with_a_carrier_fits() -> None:
    request = _request(noul(), state=State(parts=(ImagePart(data=b"png"),)))
    plan = _plan(
        request,
        modalities=frozenset({"text", "image"}),
        image_carriers=frozenset({"array"}),
        max_images=1,
        max_images_per_question=1,
    )
    assert _keys(plan) == [("q",)]


# A record with three opt-in features, two of them off, and what each adds.
_FEATURED: dict[str, Any] = {
    "opt_in_features": frozenset({"long-context", "images", "boxes", "batch"}),
    "enabled_features": frozenset({"batch"}),
    "feature_kinds": {"images": frozenset({"box"}), "boxes": frozenset({"box"})},
    "feature_modalities": {"images": frozenset({"image"})},
}


def test_does_not_fit_names_only_the_features_that_add_the_missing_kind() -> None:
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul("a"), box("b")), **_FEATURED)
    assert raised.value.check == "kinds"
    assert raised.value.observed == frozenset({"box"})
    assert raised.value.off_features == frozenset({"images", "boxes"})
    assert str(raised.value) == "kinds not supported: box; the boxes and images features add it"
    three: dict[str, Any] = {
        **_FEATURED,
        "feature_kinds": {f: frozenset({"box"}) for f in ("images", "boxes", "long-context")},
    }
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(box()), **three)
    assert str(raised.value) == (
        "kinds not supported: box; the boxes, images and long-context features add it"
    )
    one: dict[str, Any] = {**_FEATURED, "feature_kinds": {"images": frozenset({"box"})}}
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(box()), **one)
    assert raised.value.off_features == frozenset({"images"})
    assert str(raised.value) == "kinds not supported: box; the images feature adds it"


def test_does_not_fit_says_no_feature_of_the_model_adds_the_kind() -> None:
    leaf = capabilities(**_FEATURED)
    request = _request(noul("a"), box("b"), extraction("c"))
    with pytest.raises(DoesNotFit) as raised:
        _preflight(request, leaf, Linear(), Fail(), model="jev-1.13.0")
    assert raised.value.observed == frozenset({"box", "extraction"})
    assert raised.value.off_features == frozenset({"images", "boxes"})
    assert str(raised.value) == (
        "kinds not supported: box, extraction; the boxes and images features add box; "
        "no feature of model jev-1.13.0 adds extraction"
    )
    with pytest.raises(DoesNotFit) as raised:
        _preflight(_request(box("a"), extraction("b")), capabilities(), Linear(), Fail(), model="m")
    assert (
        str(raised.value)
        == "kinds not supported: box, extraction; no feature of model m adds box or extraction"
    )
    narrow = capabilities(**_FEATURED, kinds=frozenset({"noul"}))
    request = _request(choice("a"), score("b"), extraction("c"), box("d"))
    with pytest.raises(DoesNotFit) as raised:
        _preflight(request, narrow, Linear(), Fail(), model="m")
    assert str(raised.value) == (
        "kinds not supported: box, choice, extraction, score; "
        "the boxes and images features add box; "
        "no feature of model m adds choice, extraction or score"
    )
    with pytest.raises(DoesNotFit) as raised:
        _preflight(_request(extraction()), leaf, Linear(), Fail(), model="jev-1.13.0")
    assert raised.value.observed == frozenset({"extraction"})
    assert raised.value.off_features == frozenset()
    assert (
        str(raised.value)
        == "kinds not supported: extraction; no feature of model jev-1.13.0 adds it"
    )
    with pytest.raises(DoesNotFit) as raised:
        _preflight(_request(extraction()), leaf, Linear(), Fail())
    assert str(raised.value) == "kinds not supported: extraction; no feature of this model adds it"


def test_does_not_fit_names_only_the_features_that_add_the_missing_modality() -> None:
    image = State(parts=(TextPart(text="see"), ImagePart(data=b"x")))
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul(), state=image), **_FEATURED)
    assert raised.value.check == "modalities"
    assert raised.value.observed == frozenset({"image"})
    assert raised.value.limit == frozenset({"text"})
    assert raised.value.off_features == frozenset({"images"})
    assert str(raised.value) == "modalities not supported: image; the images feature adds it"
    both = State(parts=(JsonPart(value={"a": 1}), ImagePart(data=b"x")))
    with pytest.raises(DoesNotFit) as raised:
        _preflight(
            _request(noul(), state=both), capabilities(**_FEATURED), Linear(), Fail(), model="m"
        )
    assert raised.value.observed == frozenset({"image", "json"})
    assert raised.value.off_features == frozenset({"images"})
    assert str(raised.value) == (
        "modalities not supported: image, json; the images feature adds image; "
        "no feature of model m adds json"
    )


def test_limit_failure_still_names_every_off_feature() -> None:
    off = frozenset({"long-context", "images", "boxes"})
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul(), state=text_state("x" * 50)), **_FEATURED, max_input_tokens=20)
    assert raised.value.off_features == off
    assert raised.value.state_overflow
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(choice(names=("a", "b", "c"))), **_FEATURED, max_labels=2)
    assert (raised.value.check, raised.value.off_features) == ("max_labels", off)
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul("A")), **_FEATURED, key_patterns=(r"[a-z]+",))
    assert (raised.value.check, raised.value.off_features) == ("key_patterns", off)
    images: dict[str, Any] = {
        **_FEATURED,
        "modalities": frozenset({"text", "image"}),
        "max_images": UNBOUNDED,
        "max_images_per_question": UNBOUNDED,
    }
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul(), state=State(parts=(ImagePart(data=b"x"),))), **images)
    assert (raised.value.check, raised.value.off_features) == ("image_carriers", off)


def test_key_pattern_uses_fullmatch() -> None:
    patterns = (r"[a-z]+", r"[a-z_]{1,8}")
    plan = _plan(_request(noul("abc"), noul("de")), key_patterns=patterns)
    assert _keys(plan) == [("abc", "de")]
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul("abc1")), key_patterns=patterns)
    assert (raised.value.check, raised.value.observed) == ("key_patterns", ("abc1",))
    assert raised.value.limit == patterns
    assert str(raised.value) == "keys do not match key_patterns: 'abc1' does not match '[a-z]+'"
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul("abcdefghij")), key_patterns=patterns)
    assert raised.value.observed == ("abcdefghij",)
    assert raised.value.limit == patterns
    assert str(raised.value) == (
        "keys do not match key_patterns: 'abcdefghij' does not match '[a-z_]{1,8}'"
    )


def test_every_key_failing_key_patterns_is_named_in_question_order() -> None:
    patterns = (r"[a-z:]+", r"[a-z]+")
    leaf = Scripted(capabilities=capabilities(key_patterns=patterns), script=[])
    request = _request(noul("q:b"), noul("ok"), noul("Q1"), noul("q:a"))
    with pytest.raises(DoesNotFit) as raised:
        _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert raised.value.check == "key_patterns"
    assert raised.value.observed == ("q:b", "Q1", "q:a")
    assert raised.value.limit == patterns
    assert str(raised.value) == (
        "keys do not match key_patterns: 'q:b' does not match '[a-z]+', "
        "'Q1' does not match '[a-z:]+', 'q:a' does not match '[a-z]+'"
    )
    assert leaf.calls == []


def test_unknown_limit_passes_preflight_with_an_unchecked_limit_warning() -> None:
    plan = _plan(_request(noul()), max_input_tokens=None, max_questions=None)
    assert plan.outcome_warnings == (
        (
            Notice(
                code=WARNING_UNCHECKED_LIMIT,
                message="max_input_tokens is unknown, so it was not checked",
                field="max_input_tokens",
            ),
            Notice(
                code=WARNING_UNCHECKED_LIMIT,
                message="max_questions is unknown, so it was not checked",
                field="max_questions",
            ),
        ),
    )
    assert plan.response_warnings == ()


def test_unbounded_limit_is_not_checked_and_not_warned() -> None:
    estimator = Linear()
    plan = _plan(
        _request(*(noul(f"q{i}") for i in range(50)), state=text_state("x" * 10_000)),
        estimator,
        max_input_tokens=UNBOUNDED,
        max_question_tokens=UNBOUNDED,
        max_body_bytes=UNBOUNDED,
        max_questions=UNBOUNDED,
        max_total_options=UNBOUNDED,
    )
    assert len(plan.wire_requests) == 1
    assert plan.outcome_warnings == ((),)
    assert estimator.calls == []


def test_unchecked_limit_warns_only_affected_outcomes() -> None:
    request = _request(choice("c1"), noul("n1"), score("s1"), extraction("e1"))
    plan = _plan(
        request,
        kinds=frozenset({"noul", "choice", "score", "extraction"}),
        max_questions=1,
        max_labels=None,
        max_levels=None,
        min_levels=None,
        max_total_options=None,
    )
    assert _keys(plan) == [("c1",), ("n1",), ("s1",), ("e1",)]
    assert [_fields(warnings) for warnings in plan.outcome_warnings] == [
        ["max_labels", "max_total_options"],
        ["max_total_options"],
        ["max_levels", "min_levels", "max_total_options"],
        [],
    ]


_IMAGES: dict[str, Any] = {
    "modalities": frozenset({"text", "image"}),
    "image_carriers": frozenset({"parts"}),
    "max_images": UNBOUNDED,
    "max_images_per_question": UNBOUNDED,
    "max_image_bytes": UNBOUNDED,
    "max_total_image_bytes": UNBOUNDED,
    "min_image_side": 0,
    "image_media_types": frozenset({"image/png", "image/jpeg"}),
}
"""A record that takes images and checks none of the four image fields."""

_WEBP: Final = b"RIFF\x00\x00\x00\x00WEBPVP8 "


def _record(**overrides: Any) -> dict[str, Any]:
    return {**_IMAGES, **overrides}


def _with(*images: ImagePart) -> Request:
    return _request(noul(), state=State(parts=(TextPart(text="Look."), *images)))


def _refused(request: Request, **overrides: Any) -> DoesNotFit:
    with pytest.raises(DoesNotFit) as raised:
        _plan(request, **_record(**overrides))
    return raised.value


def test_image_over_max_image_bytes_does_not_fit() -> None:
    _plan(_with(ImagePart(data=png_header(size=100))), **_record(max_image_bytes=100))
    refused = _refused(
        _with(ImagePart(data=png_header()), ImagePart(data=png_header(size=101))),
        max_image_bytes=100,
        opt_in_features=frozenset({"long-context"}),
    )
    assert (refused.check, refused.observed, refused.limit) == ("max_image_bytes", 101, 100)
    assert refused.message == "image 1 is 101 bytes: max_image_bytes is 100"
    assert refused.off_features == frozenset({"long-context"})


def test_images_over_max_total_image_bytes_do_not_fit() -> None:
    pair = (ImagePart(data=png_header(size=60)), ImagePart(data=png_header(size=60)))
    _plan(_with(*pair), **_record(max_total_image_bytes=120))
    refused = _refused(_with(*pair), max_image_bytes=100, max_total_image_bytes=119)
    assert (refused.check, refused.observed, refused.limit) == ("max_total_image_bytes", 120, 119)


def test_image_under_min_image_side_does_not_fit() -> None:
    _plan(_with(ImagePart(data=png_header(280, 280))), **_record(min_image_side=280))
    refused = _refused(_with(ImagePart(data=png_header(320, 279))), min_image_side=280)
    assert (refused.check, refused.observed, refused.limit) == ("min_image_side", 279, 280)
    assert refused.message == "image 0 is 320 by 279 px: min_image_side is 280"


def test_unreadable_image_size_is_sent_with_an_unchecked_limit_warning() -> None:
    unreadable = ImagePart(data=b"\xff\xd8\xff\xda", media_type="image/jpeg")
    plan = _plan(_with(unreadable), **_record(min_image_side=280))
    [warning] = plan.outcome_warnings[0]
    assert (warning.code, warning.field) == (WARNING_UNCHECKED_LIMIT, "min_image_side")
    assert (
        warning.message
        == "the size of image 0 could not be read, so min_image_side was not checked"
    )
    two = _plan(
        _with(unreadable, ImagePart(data=png_header()), unreadable),
        **_record(min_image_side=280),
    )
    [warning] = two.outcome_warnings[0]
    assert warning.message == (
        "the sizes of images 0 and 2 could not be read, so min_image_side was not checked"
    )
    # With no minimum known above 0, nothing is checked and nothing is warned.
    assert _plan(_with(unreadable), **_IMAGES).outcome_warnings == ((),)


def test_image_media_type_outside_the_set_does_not_fit() -> None:
    refused = _refused(_with(ImagePart(data=b"GIF89a", media_type="image/gif")))
    assert (refused.check, refused.observed) == ("image_media_types", "image/gif")
    assert refused.limit == frozenset({"image/png", "image/jpeg"})
    assert refused.message == "image 0 is image/gif: image_media_types is image/jpeg, image/png"
    untyped = _refused(_with(ImagePart(data=_WEBP)))
    assert (untyped.check, untyped.observed) == ("image_media_types", "image/webp")
    unknown = _refused(_with(ImagePart(data=b"\x00\x01")))
    assert unknown.observed == "unknown"


def test_declared_media_type_outside_the_set_falls_back_to_the_signature() -> None:
    _plan(_with(ImagePart(data=png_header(), media_type="image/gif")), **_IMAGES)
    _plan(_with(ImagePart(data=b"\xff\xd8\xff\xe0", media_type="IMAGE/JPG")), **_IMAGES)
    webp = _refused(_with(ImagePart(data=_WEBP, media_type="image/gif")))
    assert webp.observed == "image/gif"


def test_unknown_image_fields_pass_preflight_with_an_unchecked_limit_warning_only_when_the_state_has_an_image() -> (  # noqa: E501
    None
):
    record: dict[str, Any] = {
        **_IMAGES,
        "max_images": None,
        "max_images_per_question": None,
        "max_image_bytes": None,
        "max_total_image_bytes": None,
        "min_image_side": None,
        "image_media_types": None,
    }
    assert _plan(_request(noul()), **record).outcome_warnings == ((),)
    plan = _plan(_with(ImagePart(data=b"any bytes at all")), **record)
    assert [_fields(w) for w in plan.outcome_warnings] == [
        [
            "max_images",
            "max_images_per_question",
            "max_image_bytes",
            "max_total_image_bytes",
            "min_image_side",
            "image_media_types",
        ]
    ]


def test_min_levels_none_is_skipped_with_a_warning() -> None:
    plan = _plan(_request(score("s", levels=1)), min_levels=None)
    assert _keys(plan) == [("s",)]
    assert _fields(plan.outcome_warnings[0]) == ["min_levels"]


def test_state_overflow_under_fail_raises_before_any_call() -> None:
    leaf = Scripted(capabilities=capabilities(max_input_tokens=100), script=[], estimator=Linear())
    request = _request(noul(), state=text_state("x" * 200))
    with pytest.raises(DoesNotFit) as raised:
        _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert raised.value.check == "max_input_tokens"
    assert raised.value.state_overflow
    assert (raised.value.observed, raised.value.limit) == (205, 100)
    assert isinstance(raised.value.__cause__, DoesNotFit)
    assert leaf.calls == []


def test_state_over_the_per_question_limit_is_a_state_overflow() -> None:
    request = _request(noul(), state=text_state("x" * 50))
    with pytest.raises(DoesNotFit) as raised:
        _plan(request, max_input_tokens=1_000, max_question_tokens=40)
    assert (raised.value.check, raised.value.state_overflow) == ("max_question_tokens", True)
    policy = Shorten(keep=10)
    plan = _plan(request, overflow=policy, max_input_tokens=1_000, max_question_tokens=40)
    assert policy.applied == [40]
    assert plan.wire_requests[0].state == text_state("x" * 10)
    assert _fields(plan.response_warnings) == ["max_question_tokens"]


def test_named_overflow_policy_is_applied_with_a_warning() -> None:
    policy = Shorten(keep=30)
    request = _request(noul("a"), noul("b"), state=text_state("x" * 200))
    plan = _plan(request, overflow=policy, max_input_tokens=100)
    assert policy.applied == [100]
    assert all(wire.state == text_state("x" * 30) for wire in plan.wire_requests)
    assert plan.response_warnings == (
        Notice(
            code=WARNING_OVERFLOW_POLICY,
            message="the state overflowed max_input_tokens; policy 'shorten' applied",
            field="max_input_tokens",
        ),
    )
    assert plan.outcome_warnings == ((),)


def test_policy_result_still_overflowing_raises() -> None:
    request = _request(noul(), state=text_state("x" * 200))
    with pytest.raises(DoesNotFit) as raised:
        _plan(request, overflow=Shorten(keep=150), max_input_tokens=100)
    assert raised.value.check == "max_input_tokens"
    assert raised.value.state_overflow
    assert raised.value.observed == 155
    assert "shorten" in str(raised.value)


def test_policy_result_is_checked_again_for_modalities() -> None:
    class ToJson:
        name = "to-json"

        def apply(self, state: State, limit: int, estimator: TokenEstimator) -> State:
            return State(parts=(JsonPart(value={"short": True}),))

    with pytest.raises(DoesNotFit) as raised:
        _plan(
            _request(noul(), state=text_state("x" * 200)),
            overflow=ToJson(),
            max_input_tokens=100,
        )
    assert raised.value.check == "modalities"


def test_state_bytes_over_the_body_limit_raise_whatever_the_policy() -> None:
    policy = Shorten(keep=1)
    request = _request(noul(), state=text_state("é" * 60))
    with pytest.raises(DoesNotFit) as raised:
        _plan(request, overflow=policy, max_body_bytes=100)
    assert (raised.value.check, raised.value.observed, raised.value.limit) == (
        "max_body_bytes",
        120,
        100,
    )
    assert raised.value.state_overflow
    assert policy.applied == []


def test_question_overflow_packs_into_wire_requests_that_repeat_the_state() -> None:
    # Each estimate is 9 (state) + 5 (overhead) + 10 per question; 40 takes two questions.
    request = _request(*(noul(f"q{i}") for i in range(5)))
    plan = _plan(request, max_input_tokens=40)
    assert _keys(plan) == [("q0", "q1"), ("q2", "q3"), ("q4",)]
    assert all(wire.state == request.state for wire in plan.wire_requests)
    assert len(plan.outcome_warnings) == len(plan.wire_requests)


def test_packing_estimates_each_candidate_prefix_whole() -> None:
    estimator = Linear()
    _plan(_request(noul("a"), noul("b"), noul("c")), estimator, max_input_tokens=40)
    assert [keys for _, keys in estimator.calls] == [
        (),
        ("a", "b", "c"),
        ("a",),
        ("a", "b"),
        ("a", "b", "c"),
        ("c",),
    ]


def test_every_key_lands_in_exactly_one_wire_request() -> None:
    request = _request(
        *(choice(f"c{i}") for i in range(4)),
        *(noul(f"n{i}") for i in range(5)),
        *(score(f"s{i}", levels=4) for i in range(3)),
    )
    plan = _plan(request, max_questions=3, max_total_options=8, max_input_tokens=60)
    keys = [key for chunk in _keys(plan) for key in chunk]
    assert keys == [question.key for question in request.questions]
    assert len(plan.wire_requests) > 4


def test_question_too_large_alone_raises_before_any_call() -> None:
    leaf = Scripted(capabilities=capabilities(max_input_tokens=30), script=[], estimator=Linear())
    big = Question(key="big", kind="noul", instruction="?")
    request = _request(noul("a"), big)

    class Heavy(Linear):
        def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
            base = super().estimate(state, questions)
            extra = 100 if any(q.key == "big" for q in questions) else 0
            return TokenEstimate(tokens=base.tokens + extra, margin=0.0)

    with pytest.raises(DoesNotFit) as raised:
        _preflight(request, leaf.capabilities, Heavy(), Fail())
    assert raised.value.check == "max_input_tokens"
    assert not raised.value.state_overflow
    assert "'big'" in str(raised.value)
    assert leaf.calls == []


def test_recounted_state_multiplies_the_estimate() -> None:
    # The state is 9 tokens + 5 overhead = 14 alone; three questions estimate 44 in one go.
    request = _request(noul("a"), noul("b"), noul("c"))
    assert _keys(_plan(request, max_input_tokens=72)) == [("a", "b", "c")]
    recounted = _plan(request, max_input_tokens=72, recounts_state_per_question=True)
    # 34 + 14 = 48 for two questions; 44 + 2 * 14 = 72 for three, which still fits.
    assert _keys(recounted) == [("a", "b", "c")]
    tighter = _plan(request, max_input_tokens=71, recounts_state_per_question=True)
    assert _keys(tighter) == [("a", "b"), ("c",)]


def test_margin_is_applied_to_each_estimate_before_summing() -> None:
    # 100 tokens with a 0.1 margin is 110 exactly, never 111 by float error.
    estimator = Linear(per_question=0, overhead=91, margin=0.1)
    request = _request(noul(), state=text_state("x" * 9))
    assert _keys(_plan(request, estimator, max_input_tokens=110)) == [("q",)]
    with pytest.raises(DoesNotFit):
        _plan(request, estimator, max_input_tokens=109)
    recounted = _plan(
        _request(noul("a"), noul("b"), state=text_state("x" * 9)),
        Linear(per_question=0, overhead=91, margin=0.1),
        max_input_tokens=220,
        recounts_state_per_question=True,
    )
    assert _keys(recounted) == [("a", "b")]


def test_per_question_tokens_include_the_state() -> None:
    estimator = Linear()
    _plan(_request(noul("a"), noul("b")), estimator, max_question_tokens=24)
    assert [keys for _, keys in estimator.calls] == [(), ("a",), ("b",), ("a", "b")]
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(noul("a")), max_question_tokens=23)
    assert (raised.value.check, raised.value.observed) == ("max_question_tokens", 24)
    assert not raised.value.state_overflow


def test_total_options_count_per_wire_request_and_noul_counts_two() -> None:
    request = _request(
        choice("c", names=("a", "b", "c")), noul("n"), score("s", levels=4), extraction("e")
    )
    plan = _plan(
        request,
        kinds=frozenset({"noul", "choice", "score", "extraction", "box"}),
        max_total_options=5,
    )
    # choice 3 + noul 2 = 5; score 4 + extraction 0 = 4.
    assert _keys(plan) == [("c", "n"), ("s", "e")]
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(choice("c", names=("a", "b", "c"))), max_total_options=2)
    assert (raised.value.check, raised.value.observed) == ("max_total_options", 3)
    boxes = _plan(
        _request(box("b1"), box("b2"), extraction("e")),
        kinds=frozenset({"extraction", "box"}),
        max_total_options=0,
    )
    assert _keys(boxes) == [("b1", "b2", "e")]


def test_body_bytes_are_measured_by_fit_including_the_questions() -> None:
    labelled = Question(
        key="k",
        kind="choice",
        instruction="ü?",
        labels=(Label(name="a", description="dé"), Label(name="b")),
    )
    image = ImagePart(data=b"12345")
    state = State(
        parts=(TextPart(text="héllo"), JsonPart(value=MappingProxyType({"a": (1, "é")})), image),
    )
    record: dict[str, Any] = {
        "modalities": frozenset({"text", "json", "image"}),
        "image_carriers": frozenset({"parts"}),
        "max_images": 1,
        "max_images_per_question": 1,
    }
    state_bytes = 6 + len('{"a":[1,"é"]}'.encode()) + len(base64.b64encode(image.data))
    question_bytes = 1 + 3 + 1 + 3 + 1
    exact = state_bytes + question_bytes
    assert _keys(_plan(_request(labelled, state=state), max_body_bytes=exact, **record)) == [("k",)]
    with pytest.raises(DoesNotFit) as raised:
        _plan(_request(labelled, state=state), max_body_bytes=exact - 1, **record)
    assert (raised.value.check, raised.value.observed) == ("max_body_bytes", exact)
    assert not raised.value.state_overflow
    two = Question(key="m", kind="choice", instruction="ü?", labels=labelled.labels)
    packed = _plan(_request(labelled, two, state=state), max_body_bytes=exact + 8, **record)
    assert _keys(packed) == [("k",), ("m",)]


def test_body_measure_refuses_a_value_that_is_not_json() -> None:
    state = State(parts=(JsonPart(value=object()),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _plan(_request(noul(), state=state), modalities=frozenset({"json"}), max_body_bytes=10)


def test_composite_checks_kinds_and_modalities_only() -> None:
    record = capabilities(
        composite=True,
        max_questions=1,
        max_input_tokens=None,
        max_labels=1,
        key_patterns=(r"z",),
    )
    _check_composite(_request(choice("a"), noul("b")), record)
    with pytest.raises(DoesNotFit) as raised:
        _check_composite(_request(box("a")), record)
    assert raised.value.check == "kinds"
    assert raised.value.off_features == frozenset()
    assert str(raised.value) == (
        "kinds not supported: box; no feature of this composite adds it; "
        "its members' own records may"
    )
    with pytest.raises(DoesNotFit) as raised:
        _check_composite(_request(box("a"), extraction("b")), record)
    assert raised.value.observed == frozenset({"box", "extraction"})
    assert str(raised.value) == (
        "kinds not supported: box, extraction; no feature of this composite adds them; "
        "its members' own records may"
    )
    with pytest.raises(DoesNotFit) as raised:
        _check_composite(_request(noul(), state=State(parts=(ImagePart(data=b"x"),))), record)
    assert raised.value.check == "modalities"
    assert str(raised.value) == (
        "modalities not supported: image; no feature of this composite adds it; "
        "its members' own records may"
    )


def test_merge_keeps_plan_order_and_leaves_failed_keys_absent() -> None:
    first, failed, last = _outcome("a", "b"), _outcome("c", error=ProviderError("x")), _outcome("d")
    yes = NoulAnswer(probability=0.7)
    answers, outcomes = _merge(
        [
            (first, {"a": yes, "b": yes}),
            (failed, None),
            (last, {"d": yes}),
        ]
    )
    assert list(answers) == ["a", "b", "d"]
    assert "c" not in answers
    assert outcomes == (first, failed, last)


def test_fit_plan_keeps_its_warnings_parallel_to_its_wire_requests() -> None:
    with pytest.raises(ValueError, match="parallel"):
        FitPlan(wire_requests=(), outcome_warnings=((),))


def test_every_wire_request_of_a_plan_carries_the_requests_options() -> None:
    request = dataclasses.replace(
        _request(noul("a"), noul("b"), noul("c")), options={"reference_date": "2026-10-21"}
    )
    plan = _plan(request, max_questions=1)
    assert len(plan.wire_requests) == 3
    assert all(w.options == {"reference_date": "2026-10-21"} for w in plan.wire_requests)
