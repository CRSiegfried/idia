"""The System One leaf: its record per model and feature, the wire body, the reading of
each kind, its provenance, its estimator, and its place in the registry."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import subprocess
import sys
import zlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import httpx2
import pytest

import idia
from idia import (
    UNBOUNDED,
    WARNING_EXTRA_ANSWER_KEY,
    WARNING_NO_SNAPSHOT,
    Box,
    BoxAnswer,
    Deadline,
    Dispatcher,
    DoesNotFit,
    Fail,
    Fault,
    ImagePart,
    JsonPart,
    JsonValue,
    Label,
    LabelAnswer,
    Leaf,
    NoulAnswer,
    Part,
    Provider,
    ProviderError,
    Question,
    Request,
    Rule,
    RuleFailure,
    State,
    TextPart,
    WireRequest,
    WireResponse,
    providers,
)
from idia.fit import _preflight
from idia.providers.system_one import (
    _MIN_IMAGE_SIDE,
    SystemOne,
    SystemOneEstimator,
)
from tests._http import FILTERED, body_key, load_cassette, replay_headers
from tests._system_one_cases import (
    BOX_PROOF,
    BOX_PROOF_REQUEST,
    CONFIGURATIONS,
    FAILURE,
    IMAGE_SIDE_PROBES,
    SIDE_PROBE,
    TEXT,
    TEXT_QUESTIONS,
    WITH_IMAGES,
    WORDING,
    Configuration,
    check_red_square,
    decode,
    load,
    png,
    record,
    side_probe,
    side_probe_path,
    wording_request,
)

Handler = Callable[[httpx2.Request], httpx2.Response]


class Wire:
    """A mock transport that answers every request with one response and keeps what it
    was sent; a recorded response's body-encoding headers are dropped, since its body is
    written again, decoded."""

    def __init__(
        self,
        body: object,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status = status
        kept = replay_headers(headers or {})
        self.headers = {"x-typesafe-request-id": "req-1", **kept}
        self.sent: list[httpx2.Request] = []
        self.client = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.sent.append(request)
        if isinstance(self.body, bytes):
            return httpx2.Response(self.status, content=self.body, headers=self.headers)
        return httpx2.Response(self.status, json=self.body, headers=self.headers)

    def leaf(self, **options: Any) -> SystemOne:
        return SystemOne(api_key="test-key", client=self.client, **options)

    @property
    def body_sent(self) -> Any:
        [request] = self.sent
        return json.loads(request.content)


def _call(leaf: SystemOne, wire_request: WireRequest) -> WireResponse:
    return asyncio.run(leaf.call(wire_request, Deadline.after(5)))


def _raised(leaf: SystemOne, wire_request: WireRequest) -> ProviderError:
    with pytest.raises(ProviderError) as raised:
        _call(leaf, wire_request)
    return raised.value


def _answering(answers: Mapping[str, object], model: str = "jev-1.14.0") -> Wire:
    return Wire({"model": model, "answers": answers, "usage": {"input_tokens": 9}})


def test_a_json_part_nested_too_deeply_never_reaches_the_wire() -> None:
    value: JsonValue = 1
    for _ in range(100_000):
        value = (value,)
    # A value nested too deep never reaches the wire: it is refused when the part is built.
    with pytest.raises(ValueError, match=r"^JsonPart\.value nests too deep$"):
        JsonPart(value=value)


def test_a_state_with_a_lone_surrogate_is_refused_as_a_provider_error() -> None:
    surrogate = State(parts=(TextPart(text="a\ud800"),))
    error = _raised(_answering({}).leaf(), _one(_NOUL, surrogate))
    assert "cannot be sent as JSON" in error.message


def _one(question: Question, state: State = TEXT) -> WireRequest:
    return WireRequest(state=state, questions=(question,))


_NOUL = Question(key="q", kind="noul", instruction="Is it so?")
_CHOICE = TEXT_QUESTIONS["choice"][0]
_SCORE = TEXT_QUESTIONS["score"][0]
_NAMED_SCORE = TEXT_QUESTIONS["score"][1]
_BOX = Question(key="b", kind="box", instruction="The red square.")
_PICTURE = State(parts=(ImagePart(data=png(), media_type="image/png"),))

# ----------------------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["jev-1.13.0", "jev-latest", "jev-1.14.0", "jev-preview"])
@pytest.mark.parametrize("long_context", [False, True])
def test_the_record_per_model_and_long_context(model: str, long_context: bool) -> None:
    leaf = SystemOne(model=model, api_key="k", long_context=long_context)
    features = frozenset({"long_context"}) if long_context else frozenset[str]()
    assert leaf.capabilities == record(model, features)
    assert leaf.capabilities.kinds == {"noul", "choice", "score"}
    assert "image" not in leaf.capabilities.modalities


@pytest.mark.parametrize("model", ["jev-1.14.0", "jev-preview"])
def test_images_add_box_the_image_modality_and_the_placeholder(model: str) -> None:
    leaf = SystemOne(model=model, api_key="k", images=True)
    record_ = leaf.capabilities
    assert record_ == record(model, frozenset({"images"}))
    assert record_.kinds == {"noul", "choice", "score", "box"}
    assert record_.modalities == {"text", "json", "image"}
    assert record_.image_carriers == {"placeholder"}
    assert record_.max_images is UNBOUNDED
    assert record_.confidence["box"] == "undocumented"
    assert record_.enabled_features == {"images"}


def test_enabling_long_context_raises_max_input_tokens() -> None:
    short = SystemOne(api_key="k").capabilities
    long = SystemOne(api_key="k", long_context=True).capabilities
    assert (short.max_input_tokens, short.max_question_tokens) == (64_000, 32_000)
    assert (long.max_input_tokens, long.max_question_tokens) == (262_000, 32_000)
    assert long.enabled_features == {"long_context"}
    assert short.enabled_features == frozenset()


def test_the_record_states_every_field() -> None:
    capabilities = SystemOne(api_key="k").capabilities
    assert capabilities.max_questions is UNBOUNDED
    assert capabilities.max_total_options is UNBOUNDED
    assert (capabilities.max_labels, capabilities.max_levels, capabilities.min_levels) == (
        255,
        10,
        1,
    )
    assert capabilities.max_body_bytes == 33_554_432
    assert capabilities.returns_refusal is False
    assert capabilities.calibrated
    assert capabilities.snapshot_pinnable
    assert capabilities.truncates_silently is False
    assert "noul" not in capabilities.confidence
    assert capabilities.opt_in_features == {"long_context", "images"}


def test_an_unknown_model_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="jev-9"):
        SystemOne(model="jev-9", api_key="k")


@pytest.mark.parametrize("model", ["jev-1.13.0", "jev-latest"])
def test_images_on_1_13_add_the_image_modality_but_not_box(model: str) -> None:
    capabilities = SystemOne(model=model, api_key="k", images=True).capabilities
    assert capabilities == record(model, frozenset({"images"}))
    assert capabilities.kinds == {"noul", "choice", "score"}
    assert capabilities.modalities == {"text", "json", "image"}
    assert "box" not in capabilities.confidence
    assert capabilities.feature_kinds == {}
    assert capabilities.feature_modalities == {"images": {"image"}}


_FEATURE_SETS = [
    frozenset[str](),
    frozenset({"images"}),
    frozenset({"long_context"}),
    frozenset({"images", "long_context"}),
]


@pytest.mark.parametrize("model", ["jev-1.13.0", "jev-latest", "jev-1.14.0", "jev-preview"])
def test_the_feature_maps_are_the_same_whatever_is_enabled(model: str) -> None:
    built = [
        SystemOne(
            model=model,  # pyright: ignore[reportArgumentType]
            api_key="k",
            images="images" in features,
            long_context="long_context" in features,
        ).capabilities
        for features in _FEATURE_SETS
    ]
    boxes = model in ("jev-1.14.0", "jev-preview")
    for capabilities in built:
        assert capabilities.feature_kinds == ({"images": {"box"}} if boxes else {})
        assert capabilities.feature_modalities == {"images": {"image"}}
        assert "long_context" not in capabilities.feature_kinds
        assert "long_context" not in capabilities.feature_modalities


def _refused(leaf: SystemOne, request: Request) -> DoesNotFit:
    with pytest.raises(DoesNotFit) as raised:
        asyncio.run(Dispatcher(leaf).answer(request))
    return raised.value


def test_a_box_on_1_13_with_images_on_says_no_feature_of_the_model_adds_it() -> None:
    box = Question(key="b", kind="box", instruction="Where is the sign?")
    noul = Question(key="n", kind="noul", instruction="Is it a sign?")
    request = Request(state=TEXT, questions=(noul, box), deadline=Deadline.after(5))
    refused = _refused(SystemOne(model="jev-1.13.0", api_key="k", images=True), request)
    assert str(refused) == "kinds not supported: box; no feature of model jev-1.13.0 adds it"
    assert refused.observed == frozenset({"box"})
    assert refused.off_features == frozenset()


def test_a_box_on_1_14_without_images_names_the_images_feature() -> None:
    box = Question(key="b", kind="box", instruction="Where is the sign?")
    request = Request(state=TEXT, questions=(box,), deadline=Deadline.after(5))
    refused = _refused(SystemOne(model="jev-1.14.0", api_key="k"), request)
    assert str(refused) == "kinds not supported: box; the images feature adds it"
    assert refused.off_features == frozenset({"images"})


def test_an_image_without_the_images_feature_names_only_images() -> None:
    noul = Question(key="n", kind="noul", instruction="Is it a sign?")
    state = State(parts=(TextPart(text="See "), ImagePart(data=png(), media_type="image/png")))
    request = Request(state=state, questions=(noul,), deadline=Deadline.after(5))
    refused = _refused(SystemOne(model="jev-1.13.0", api_key="k"), request)
    assert str(refused) == "modalities not supported: image; the images feature adds it"
    assert refused.observed == frozenset({"image"})
    assert refused.off_features == frozenset({"images"})


def test_1_13_answered_a_question_about_an_image() -> None:
    probe = Configuration("jev-1.13.0", frozenset({"images"}))
    cassette = load(probe, "probe-noul")
    response = cassette["response"]
    wire = Wire(response["body"], status=response["status"], headers=response["headers"])
    wire_request = decode(cassette["wire_request"])
    answers = _call(wire.leaf(**probe.options()), wire_request).answers
    assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
    assert wire.sent[0].headers["typesafe-danger-zone-images"] == "true"
    assert set(answers) == {"noul_red", "noul_green"}
    assert all(isinstance(answer, NoulAnswer) for answer in answers.values())


def test_the_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    wire = _answering({"q": {"type": "noul", "noul": 0.5}})
    leaf = SystemOne(client=wire.client)
    _call(leaf, _one(_NOUL))
    assert wire.sent[0].headers["authorization"] == "Bearer from-env"
    assert "from-env" not in repr(leaf)


def test_no_key_is_refused_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        SystemOne()


def test_a_caller_may_pass_a_changed_record() -> None:
    changed = dataclasses.replace(SystemOne(api_key="k").capabilities, max_labels=100)
    assert SystemOne(api_key="k", capabilities=changed).capabilities is changed


def test_it_is_a_leaf_and_not_a_provider_until_wrapped() -> None:
    leaf = SystemOne(api_key="k")
    assert isinstance(leaf, Leaf)
    assert not isinstance(leaf, Provider)
    assert isinstance(Dispatcher(leaf), Provider)
    assert (leaf.name, leaf.model) == ("system-one", "jev-1.14.0")
    with pytest.raises(AttributeError):
        leaf.capabilities = leaf.capabilities  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]


# ----------------------------------------------------------------------------------------
# The wire body
# ----------------------------------------------------------------------------------------


def test_the_body_per_kind() -> None:
    questions = (
        _NOUL,
        _CHOICE,
        _SCORE,
        _NAMED_SCORE,
    )
    wire = _answering({})
    _call(wire.leaf(), WireRequest(state=TEXT, questions=questions))
    body = wire.body_sent
    assert list(body) == ["model", "state", "questions"]
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == TEXT.parts[0].text  # type: ignore[union-attr]
    assert list(body["questions"]) == ["q", "choice_weather", "score_rain", "score_mood"]
    assert body["questions"] == {
        "q": {"type": "noul", "instructions": "Is it so?"},
        "choice_weather": {
            "type": "choice",
            "instructions": "The weather?",
            "criteria": {"rain": "It rained.", "sun": None, "snow": "It snowed."},
        },
        "score_rain": {
            "type": "score",
            "instructions": "How hard?",
            "criteria": ["No rain", "Light rain", "Heavy rain"],
        },
        "score_mood": {
            "type": "score",
            "instructions": "The mood?",
            "criteria": ["low", "mid", "high"],
        },
    }
    assert list(body["questions"]["choice_weather"]["criteria"]) == ["rain", "sun", "snow"]


def test_the_request_line_and_headers() -> None:
    wire = _answering({})
    _call(wire.leaf(), _one(_NOUL))
    [request] = wire.sent
    assert request.method == "POST"
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"] == f"idia/{idia.__version__}"
    assert "typesafe-danger-zone-images" not in request.headers
    assert "typesafe-danger-zone-long-context" not in request.headers
    assert "content-encoding" not in request.headers


def test_each_feature_sends_its_header() -> None:
    wire = _answering({}, "jev-1.14.0")
    _call(wire.leaf(model="jev-1.14.0", images=True, long_context=True), _one(_NOUL))
    headers = wire.sent[0].headers
    assert headers["typesafe-danger-zone-images"] == "true"
    assert headers["typesafe-danger-zone-long-context"] == "true"


def test_a_base_url_may_be_given() -> None:
    wire = _answering({})
    _call(wire.leaf(base_url="https://staging.test/"), _one(_NOUL))
    assert str(wire.sent[0].url) == "https://staging.test/v1/systemone"


def test_several_parts_are_an_array_in_order_and_json_is_its_value() -> None:
    state = State(parts=(TextPart(text="first"), JsonPart(value={"order": {"charges": (1, 2)}})))
    wire = _answering({})
    _call(wire.leaf(), _one(_NOUL, state))
    assert wire.body_sent["state"] == ["first", {"order": {"charges": [1, 2]}}]


def test_a_json_part_alone_is_its_own_value() -> None:
    wire = _answering({})
    _call(wire.leaf(), _one(_NOUL, State(parts=(JsonPart(value=(1, "a")),))))
    assert wire.body_sent["state"] == [1, "a"]


def test_an_image_is_a_placeholder_with_its_asset() -> None:
    data = png()
    state = State(parts=(TextPart(text="Look:"), ImagePart(data=data, media_type="image/png")))
    wire = _answering({}, "jev-1.14.0")
    response = _call(wire.leaf(model="jev-1.14.0", images=True), _one(_BOX, state))
    body = wire.body_sent
    assert body["state"] == ["Look:", "<image-0>"]
    assert body["questions"] == {"b": {"type": "box", "instructions": "The red square."}}
    assert body["assets"] == {
        "<image-0>": {
            "type": "image",
            "format": "png",
            "data_base64": base64.b64encode(data).decode(),
        }
    }
    assert response.prompt_sent is not None
    assert isinstance(response.prompt_sent, Mapping)
    assert response.prompt_sent["assets"] == {
        "<image-0>": {"type": "image", "format": "png", "bytes": len(data)}
    }
    assert response.prompt_sent["state"] == ("Look:", "<image-0>")


@pytest.mark.parametrize(
    ("data", "media_type", "expected"),
    [
        (b"\xff\xd8\xff\xe0rest", "image/jpeg", "jpeg"),
        (b"\xff\xd8\xff\xe0rest", "IMAGE/JPG", "jpeg"),
        (b"\xff\xd8\xff\xe0rest", None, "jpeg"),
        (png(), None, "png"),
        (png(), "application/octet-stream", "png"),
    ],
)
def test_the_image_format(data: bytes, media_type: str | None, expected: str) -> None:
    wire = _answering({}, "jev-1.14.0")
    state = State(parts=(ImagePart(data=data, media_type=media_type),))
    _call(wire.leaf(model="jev-1.14.0", images=True), _one(_BOX, state))
    assert wire.body_sent["assets"]["<image-0>"]["format"] == expected


def test_a_leaf_called_past_fit_sends_an_image_of_another_format_for_the_wire_to_judge() -> None:
    wire = _answering({}, "jev-1.14.0")
    state = State(parts=(ImagePart(data=b"GIF89a", media_type="image/gif"),))
    _call(wire.leaf(model="jev-1.14.0", images=True), _one(_BOX, state))
    assert wire.body_sent["assets"]["<image-0>"]["format"] == "gif"


def test_the_body_is_byte_stable() -> None:
    wire = _answering({})
    leaf = wire.leaf()
    request = WireRequest(state=TEXT, questions=(_NOUL, _CHOICE, _SCORE))

    async def twice() -> None:  # one loop, as an injected client's calls must be
        for _ in range(2):
            await leaf.call(request, Deadline.after(5))

    asyncio.run(twice())
    assert wire.sent[0].content == wire.sent[1].content


def test_the_prompt_sent_is_the_body_without_assets() -> None:
    wire = _answering({"q": {"type": "noul", "noul": 0.5}})
    response = _call(wire.leaf(), _one(_NOUL))
    assert response.prompt_sent == {
        "model": "jev-1.14.0",
        "state": TEXT.parts[0].text,  # type: ignore[union-attr]
        "questions": {"q": {"type": "noul", "instructions": "Is it so?"}},
    }


# ----------------------------------------------------------------------------------------
# Reading the answers
# ----------------------------------------------------------------------------------------


def test_a_noul_is_its_one_probability() -> None:
    response = _call(_answering({"q": {"type": "noul", "noul": 0.37}}).leaf(), _one(_NOUL))
    assert response.answers == {"q": NoulAnswer(probability=0.37)}


def test_a_choice_in_label_order_with_its_choice_and_confidence() -> None:
    answer = {
        "type": "choice",
        "choice": "rain",
        "confidence": 0.85,
        "probabilities": {"snow": 0.02, "rain": 0.9, "sun": 0.08},
    }
    response = _call(_answering({"choice_weather": answer}).leaf(), _one(_CHOICE))
    assert response.answers["choice_weather"] == LabelAnswer(
        probabilities=(("rain", 0.9), ("sun", 0.08), ("snow", 0.02)),
        confidence=0.85,
        chosen="rain",
    )


def test_a_score_with_its_expected_level_and_the_legend_echoed() -> None:
    answer = {
        "type": "score",
        "score": 1.83,
        "confidence": 0.81,
        "legend": {"2": "Heavy rain", "0": "No rain", "1": "Light rain"},
        "probabilities": {"0": 0.02, "1": 0.13, "2": 0.85},
    }
    response = _call(_answering({"score_rain": answer}).leaf(), _one(_SCORE))
    assert response.answers["score_rain"] == LabelAnswer(
        probabilities=(("none", 0.02), ("light", 0.13), ("heavy", 0.85)),
        confidence=0.81,
        expected_level=1.83,
        echo=("No rain", "Light rain", "Heavy rain"),
        echo_of="descriptions",
    )


def test_a_level_with_no_description_is_echoed_by_its_name() -> None:
    answer = {
        "type": "score",
        "score": 0.5,
        "confidence": 0.4,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
    }
    response = _call(_answering({"score_mood": answer}).leaf(), _one(_NAMED_SCORE))
    mapped = response.answers["score_mood"]
    assert isinstance(mapped, LabelAnswer)
    assert mapped.echo == ("low", "mid", "high")


def test_a_legend_that_differs_from_the_levels_is_a_fault() -> None:
    answer = {
        "type": "score",
        "score": 0.5,
        "confidence": 0.4,
        "legend": {"0": "low", "1": "mid", "2": "top", "x": "odd"},
        "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
    }
    response = _call(_answering({"score_mood": answer}).leaf(), _one(_NAMED_SCORE))
    fault = response.answers["score_mood"]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.echo_mismatch]
    assert fault.failures[0].observed == "low,mid,top,odd"


def test_a_box_and_a_null_confidence() -> None:
    box = {"top": 0.25, "bottom": 0.5, "left": 0.2, "right": 0.6}
    wire = _answering(
        {
            "b": {"type": "box", "box": box, "confidence": 0.9},
            "c": {"type": "box", "box": box, "confidence": None},
        },
        "jev-1.14.0",
    )
    second = Question(key="c", kind="box", instruction="The blue square.")
    response = _call(
        wire.leaf(model="jev-1.14.0", images=True),
        WireRequest(state=_PICTURE, questions=(_BOX, second)),
    )
    region = Box(top=0.25, left=0.2, bottom=0.5, right=0.6)
    assert response.answers == {
        "b": BoxAnswer(region=region, confidence=0.9),
        "c": BoxAnswer(region=region, confidence=None),
    }


def test_a_missing_label_stays_missing_and_validation_faults_it() -> None:
    answer = {
        "type": "choice",
        "choice": "rain",
        "confidence": 0.85,
        "probabilities": {"rain": 0.92, "snow": 0.08},
    }
    response = _call(_answering({"choice_weather": answer}).leaf(), _one(_CHOICE))
    fault = response.answers["choice_weather"]
    assert isinstance(fault, Fault)
    assert fault.failures[0].rule == Rule.labels_mismatch
    assert fault.failures[0].observed == "rain,snow"
    assert isinstance(fault.mapped, LabelAnswer)


def test_a_label_the_wire_added_comes_after_the_asked_ones() -> None:
    answer = {
        "type": "choice",
        "choice": "rain",
        "confidence": 0.85,
        "probabilities": {"hail": 0.0, "rain": 0.9, "sun": 0.08, "snow": 0.02},
    }
    response = _call(_answering({"choice_weather": answer}).leaf(), _one(_CHOICE))
    fault = response.answers["choice_weather"]
    assert isinstance(fault, Fault)
    assert fault.failures[0].observed == "rain,sun,snow,hail"


def test_a_missing_answer_is_a_fault_and_an_extra_key_is_dropped_with_a_warning() -> None:
    wire = _answering({"other": {"type": "noul", "noul": 0.5}})
    response = _call(wire.leaf(), _one(_NOUL))
    fault = response.answers["q"]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.missing_key]
    assert set(response.answers) == {"q"}
    assert [w.code for w in response.warnings] == [WARNING_EXTRA_ANSWER_KEY]


@pytest.mark.parametrize(
    ("answer", "observed"),
    [
        ({"type": "choice", "choice": "a", "probabilities": {}}, "choice"),
        ({"noul": 0.5}, "None"),
        ({"type": "noul", "noul": "0.5"}, "str"),
        ({"type": "noul", "noul": True}, "bool"),
        ({"type": "noul"}, "NoneType"),
        ([0.5], "tuple"),
    ],
)
def test_a_wrong_shape_for_one_question_is_its_fault(answer: object, observed: str) -> None:
    sibling = {
        "type": "choice",
        "choice": "rain",
        "confidence": 0.85,
        "probabilities": {"rain": 0.9, "sun": 0.08, "snow": 0.02},
    }
    wire = _answering({"q": answer, "choice_weather": sibling})
    response = _call(wire.leaf(), WireRequest(state=TEXT, questions=(_NOUL, _CHOICE)))
    failure = RuleFailure(rule=Rule.kind_shape, observed=observed, bound=None)
    assert response.answers["q"] == Fault(key="q", failures=(failure,), mapped=None)
    assert isinstance(response.answers["choice_weather"], LabelAnswer)


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": 3, "confidence": 0.1, "probabilities": {}},
        {"type": "choice", "choice": "a", "confidence": "high", "probabilities": {}},
        {"type": "choice", "choice": "a", "confidence": 0.1, "probabilities": []},
        {"type": "choice", "choice": "a", "probabilities": {"rain": None}},
    ],
)
def test_a_malformed_choice_is_a_kind_shape_fault(answer: object) -> None:
    response = _call(_answering({"choice_weather": answer}).leaf(), _one(_CHOICE))
    fault = response.answers["choice_weather"]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.kind_shape]


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "score", "score": 1, "legend": {"0": 1}, "probabilities": {"0": 1.0}},
        {"type": "score", "score": 1, "legend": [], "probabilities": {"0": 1.0}},
        {"type": "score", "score": None, "legend": {}, "probabilities": {"0": 1.0}},
        {"type": "box", "box": {"top": 0.1}},
    ],
)
def test_a_malformed_score_or_box_is_a_kind_shape_fault(answer: object) -> None:
    question = _SCORE if answer["type"] == "score" else _BOX  # type: ignore[index]
    wire = _answering({question.key: answer}, "jev-1.14.0")
    leaf = wire.leaf(model="jev-1.14.0", images=True)
    response = _call(leaf, _one(question, _PICTURE))
    fault = response.answers[question.key]
    assert isinstance(fault, Fault)
    assert [f.rule for f in fault.failures] == [Rule.kind_shape]


# ----------------------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------------------


def test_a_pinned_model_is_the_snapshot() -> None:
    wire = Wire(
        {
            "model": "jev-1.13.0",
            "answers": {"q": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 120, "output_tokens": 3},
        }
    )
    response = _call(wire.leaf(model="jev-latest"), _one(_NOUL))
    assert response.model_reported == "jev-1.13.0"
    assert response.snapshot == "jev-1.13.0"
    assert response.elicitation == "passthrough"
    assert response.calibrated is True
    assert response.request_id == "req-1"
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (120, 3)
    assert response.usage.raw_usage == {"input_tokens": 120, "output_tokens": 3}
    assert response.raw_body == {
        "model": "jev-1.13.0",
        "answers": {"q": {"type": "noul", "noul": 0.5}},
        "usage": {"input_tokens": 120, "output_tokens": 3},
    }
    assert response.warnings == ()


@pytest.mark.parametrize(
    "reported", ["jev-latest", "jev-1.13", "jev-1.13.0-rc1", "jev-1.13.0\n", "jev-\u0661.13.0"]
)
def test_an_alias_reported_is_no_snapshot_and_a_warning(reported: str) -> None:
    wire = _answering({"q": {"type": "noul", "noul": 0.5}}, reported)
    response = _call(wire.leaf(), _one(_NOUL))
    assert response.model_reported == reported
    assert response.snapshot is None
    assert [w.code for w in response.warnings] == [WARNING_NO_SNAPSHOT]


def test_usage_counts_that_are_not_integers_are_none_and_kept_raw() -> None:
    wire = Wire({"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": "9"}})
    response = _call(wire.leaf(), _one(_NOUL))
    assert response.usage is not None
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.raw_usage == {"input_tokens": "9"}


@pytest.mark.parametrize(
    "body",
    [
        {"answers": {}, "usage": {}},
        {"model": "", "answers": {}, "usage": {}},
        {"model": 1.13, "answers": {}, "usage": {}},
        {"model": "jev-1.13.0", "answers": {}},
        {"model": "jev-1.13.0", "answers": {}, "usage": []},
    ],
)
def test_a_body_failing_the_top_level_schema_is_not_retryable(body: object) -> None:
    error = _raised(Wire(body).leaf(), _one(_NOUL))
    assert not error.is_retryable
    assert error.status_code == 200
    assert error.request_id == "req-1"
    assert "schema" in error.message


def test_an_error_status_carries_the_request_id() -> None:
    wire = Wire({"detail": "nope"}, status=401, headers={"x-typesafe-request-id": "r-401"})
    error = _raised(wire.leaf(), _one(_NOUL))
    assert (error.status_code, error.request_id, error.is_retryable) == (401, "r-401", False)


def test_every_429_is_retryable_until_a_quota_body_is_known() -> None:
    wire = Wire({"error": "quota exceeded"}, status=429, headers={"retry-after-ms": "250"})
    error = _raised(wire.leaf(), _one(_NOUL))
    assert error.is_retryable
    assert error.retry_after == 0.25


# ----------------------------------------------------------------------------------------
# The estimator
# ----------------------------------------------------------------------------------------


def test_the_estimate_for_text_and_questions() -> None:
    state = State(parts=(TextPart(text="x" * 58),))  # 60 characters of JSON: 10 tokens
    question = Question(key="k", kind="noul", instruction="y" * 17)
    # {"k": {"type": "noul", "instructions": "yyy..."}} is 60 characters: 15 tokens
    estimate = SystemOneEstimator().estimate(state, [question])
    assert (estimate.tokens, estimate.margin) == (260 + 10 + 15, 0.1)


def test_the_state_alone_costs_the_overhead_and_its_characters() -> None:
    estimate = SystemOne(api_key="k").estimator.estimate(State(parts=(TextPart(text="abcd"),)), [])
    assert estimate.tokens == 260 + 1


def test_each_image_costs_its_figure() -> None:
    state = State(parts=(TextPart(text="abcd"), ImagePart(data=png())))
    estimate = SystemOneEstimator(image_tokens=700, margin=0.2).estimate(state, [])
    # ["abcd", "<image-0>"] is 21 characters: 4 tokens
    assert (estimate.tokens, estimate.margin) == (260 + 4 + 700, 0.2)


def test_a_json_state_is_measured_as_json() -> None:
    state = State(parts=(JsonPart(value={"a": 1}),))
    assert SystemOneEstimator().estimate(state, []).tokens == 260 + 2


def test_a_read_only_mapping_in_the_state_is_sent_as_an_object() -> None:
    state = State(parts=(JsonPart(value=MappingProxyType({"a": 1})),))
    wire = _answering({})
    _call(wire.leaf(), _one(_NOUL, state))
    assert wire.body_sent["state"] == {"a": 1}
    assert SystemOneEstimator().estimate(state, []).tokens == 260 + 2


def test_a_non_json_value_in_the_state_raises() -> None:
    state = State(parts=(JsonPart(value={"a": object()}),))  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="object"):
        SystemOneEstimator().estimate(state, [])


# ----------------------------------------------------------------------------------------
# The cassettes, replayed
# ----------------------------------------------------------------------------------------


def _replay(configuration: Configuration, name: str) -> tuple[Wire, SystemOne, WireRequest]:
    cassette = load(configuration, name)
    response = cassette["response"]
    wire = Wire(response["body"], status=response["status"], headers=response["headers"])
    leaf = wire.leaf(**configuration.options())
    wire_request = decode(cassette["wire_request"])
    assert isinstance(wire_request, WireRequest)
    return wire, leaf, wire_request


_CASES = [
    pytest.param(configuration, name, wire_request, id=f"{configuration.model}-{name}")
    for configuration in CONFIGURATIONS
    for name, wire_request in [
        *((kind, configuration.wire_request(kind)) for kind in configuration.kinds),
        *configuration.singles().items(),
    ]
]


@pytest.mark.parametrize(("configuration", "name", "asked"), _CASES)
def test_each_cassette_is_the_body_sent_and_its_answers_as_recorded(
    configuration: Configuration, name: str, asked: WireRequest
) -> None:
    cassette = load(configuration, name)
    wire, leaf, wire_request = _replay(configuration, name)
    assert wire_request == asked
    response = _call(leaf, wire_request)
    assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
    assert response.answers == decode(cassette["expected"])
    assert not any(isinstance(answer, Fault) for answer in response.answers.values())
    assert response.snapshot == configuration.model
    assert cassette["request"]["headers"].get("authorization", FILTERED) == FILTERED


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=lambda c: c.model)
def test_the_wording_cassette_sends_criteria_and_the_structured_instruction(
    configuration: Configuration,
) -> None:
    cassette = load(configuration, WORDING)
    wire, leaf, decoded = _replay(configuration, WORDING)
    # The cassette's wire request is stored with sorted keys, as every canonical form is,
    # so the call is the request as asked, its instruction's keys in the order sent.
    wire_request = wording_request()
    assert decoded == wire_request
    response = _call(leaf, wire_request)
    assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
    assert response.answers == decode(cassette["expected"])
    assert all(isinstance(answer, NoulAnswer) for answer in response.answers.values())
    described, structured = wire_request.questions
    sent = wire.body_sent["questions"]
    assert sent["described"]["criteria"] == {
        "true": described.labels[0].description,
        "false": described.labels[1].description,
    }
    assert sent["structured"]["instructions"] == structured.instruction
    estimate = leaf.estimator.estimate(wire_request.state, wire_request.questions)
    used = cassette["response"]["body"]["usage"]["input_tokens"]
    assert used <= estimate.tokens * (1 + estimate.margin)


def test_a_request_over_the_token_limit_is_refused_by_the_wire() -> None:
    _, leaf, wire_request = _replay(CONFIGURATIONS[0], "over-limit")
    error = _raised(leaf, wire_request)
    assert error.status_code == 400
    assert not error.is_retryable
    assert "max_tokens_exceeded" in error.message


def test_the_estimate_covers_the_recorded_usage() -> None:
    for configuration in CONFIGURATIONS:
        for name in [*configuration.kinds, *configuration.singles()]:
            cassette = load(configuration, name)
            wire_request = decode(cassette["wire_request"])
            estimate = SystemOne(api_key="k").estimator.estimate(
                wire_request.state, wire_request.questions
            )
            used = cassette["response"]["body"]["usage"]["input_tokens"]
            assert used <= estimate.tokens * (1 + estimate.margin), (configuration, name)


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=lambda c: c.model)
def test_eleven_levels_are_refused_by_the_wire(configuration: Configuration) -> None:
    _, leaf, wire_request = _replay(configuration, FAILURE)
    assert len(wire_request.questions[0].labels) == 11
    error = _raised(leaf, wire_request)
    assert error.status_code is not None
    assert error.status_code >= 400
    assert not error.is_retryable
    assert error.request_id


# ----------------------------------------------------------------------------------------
# The registry and the extra
# ----------------------------------------------------------------------------------------


def test_the_registry_names_the_leaf_and_its_extra() -> None:
    entry = providers._REGISTRY["system-one"]
    assert entry.target == "idia.providers.system_one:SystemOne"
    assert entry.extra == "system-one"
    assert providers._EXTRAS["system-one"] == ("httpx2",)


def test_provider_builds_and_wraps_it() -> None:
    wire = _answering({"q": {"type": "noul", "noul": 0.6}})
    provider = idia.provider("system-one", api_key="k", client=wire.client)
    assert isinstance(provider, Dispatcher)
    request = Request(state=TEXT, questions=(_NOUL,), deadline=Deadline.after(5))
    response = asyncio.run(provider.answer(request))
    assert response.answers == {"q": NoulAnswer(probability=0.6)}
    [outcome] = response.outcomes
    assert (outcome.provider, outcome.model_requested, outcome.snapshot) == (
        "system-one",
        "jev-1.14.0",
        "jev-1.14.0",
    )


_WITHOUT_HTTPX2 = """
import sys
sys.modules["httpx2"] = None  # a blocked import raises ModuleNotFoundError
import idia
assert "idia.providers.http" not in sys.modules
try:
    idia.provider("system-one", api_key="k")
except ModuleNotFoundError as error:
    assert error.name == "httpx2", error.name
    print(error)
"""


def test_without_httpx2_idia_imports_and_the_leaf_asks_for_its_extra() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _WITHOUT_HTTPX2], capture_output=True, text=True, check=True
    )
    assert 'pip install "idia[system-one]"' in done.stdout


def test_a_missing_label_that_also_breaks_the_sum_faults_both_rules() -> None:
    question = Question(
        key="c", kind="choice", instruction="?", labels=(Label(name="a"), Label(name="b"))
    )
    answer = {"type": "choice", "choice": "a", "confidence": 0.2, "probabilities": {"a": 0.6}}
    response = _call(_answering({"c": answer}).leaf(), _one(question))
    fault = response.answers["c"]
    assert isinstance(fault, Fault)
    assert {f.rule for f in fault.failures} == {Rule.labels_mismatch, Rule.sum_tolerance}


# ----------------------------------------------------------------------------------------
# Refusals before sending, and readings that must not raise
# ----------------------------------------------------------------------------------------


def _two_images() -> State:
    image = ImagePart(data=png(), media_type="image/png")
    return State(parts=(image, TextPart(text="and"), image))


def test_a_box_over_a_state_with_two_images_is_refused_before_sending() -> None:
    wire = _answering({}, "jev-1.14.0")
    error = _raised(wire.leaf(model="jev-1.14.0", images=True), _one(_BOX, _two_images()))
    assert not error.is_retryable
    assert "box" in error.message
    assert "2" in error.message
    assert wire.sent == []


def test_other_kinds_may_see_two_images() -> None:
    wire = _answering({"q": {"type": "noul", "noul": 0.5}}, "jev-1.14.0")
    response = _call(wire.leaf(model="jev-1.14.0", images=True), _one(_NOUL, _two_images()))
    assert response.answers == {"q": NoulAnswer(probability=0.5)}
    assert list(wire.body_sent["assets"]) == ["<image-0>", "<image-1>"]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {"when": object()}], ids=["nan", "inf", "object"]
)
def test_a_state_that_is_not_json_is_refused_before_sending(value: object) -> None:
    wire = _answering({})
    state = State(parts=(JsonPart(value=value),))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    error = _raised(wire.leaf(), _one(_NOUL, state))
    assert not error.is_retryable
    assert "JSON" in error.message
    assert wire.sent == []


def test_a_number_too_large_for_a_float_is_a_kind_shape_fault() -> None:
    wire = Wire(
        b'{"model": "jev-1.13.0", "usage": {}, "answers": {"q": {"type": "noul", "noul": 1'
        + b"0" * 400
        + b"}}}"
    )
    fault = _call(wire.leaf(), _one(_NOUL)).answers["q"]
    assert isinstance(fault, Fault)
    assert [(f.rule, f.observed) for f in fault.failures] == [(Rule.kind_shape, "int beyond float")]


def test_a_legend_key_of_other_digits_sorts_last_and_faults_the_echo() -> None:
    answer = {
        "type": "score",
        "score": 0.5,
        "confidence": 0.4,
        "legend": {"0": "low", "\u00b2": "odd", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
    }
    response = _call(_answering({"score_mood": answer}).leaf(), _one(_NAMED_SCORE))
    fault = response.answers["score_mood"]
    assert isinstance(fault, Fault)
    assert fault.failures[0].observed == "low,mid,high,odd"


def test_a_legend_key_too_long_to_be_an_index_sorts_last() -> None:
    long_key = "9" * 5000
    answer = {
        "type": "score",
        "score": 0.5,
        "confidence": 0.4,
        "legend": {"0": "low", long_key: "odd", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
    }
    response = _call(_answering({"score_mood": answer}).leaf(), _one(_NAMED_SCORE))
    fault = response.answers["score_mood"]
    assert isinstance(fault, Fault)
    assert fault.failures[0].observed == "low,mid,high,odd"


@pytest.mark.parametrize(
    "body",
    [{"model": "jev-1.13.0", "usage": {}}, {"model": "jev-1.13.0", "usage": {}, "answers": [1]}],
)
def test_a_body_with_no_answers_object_is_not_retryable(body: object) -> None:
    error = _raised(Wire(body).leaf(), _one(_NOUL))
    assert not error.is_retryable
    assert "answers" in error.message


def test_box_is_answered_through_the_provider_as_recorded() -> None:
    """The box proof, recorded 2026-10-22 through ``idia.provider``: one box over the
    test card, answered on the red square by a pinned snapshot."""
    cassette = load(WITH_IMAGES, BOX_PROOF)
    response = cassette["response"]
    wire = Wire(response["body"], status=response["status"], headers=response["headers"])
    assert decode(cassette["wire_request"]) == BOX_PROOF_REQUEST
    provider = idia.provider(
        "system-one", model="jev-1.14.0", images=True, api_key="k", client=wire.client
    )
    request = Request(
        state=BOX_PROOF_REQUEST.state,
        questions=BOX_PROOF_REQUEST.questions,
        deadline=Deadline.after(5),
    )
    answered = asyncio.run(provider.answer(request))
    assert body_key(wire.body_sent) == body_key(cassette["request"]["body"])
    assert answered.answers == decode(cassette["expected"])
    check_red_square(answered)


def test_box_without_images_does_not_fit_before_any_call() -> None:
    wire = _answering({}, "jev-1.14.0")
    provider = idia.provider("system-one", model="jev-1.14.0", api_key="k", client=wire.client)
    request = Request(
        state=BOX_PROOF_REQUEST.state,
        questions=BOX_PROOF_REQUEST.questions,
        deadline=Deadline.after(5),
    )
    with pytest.raises(DoesNotFit) as refused:
        asyncio.run(provider.answer(request))
    assert refused.value.check == "kinds"
    assert refused.value.observed == frozenset({"box"})
    assert refused.value.off_features == frozenset({"images"})
    assert wire.sent == []


# ----------------------------------------------------------------------------------------
# The minimum image side
# ----------------------------------------------------------------------------------------


def _pixels(wire_request: WireRequest) -> WireRequest:
    """``wire_request`` with each PNG's bytes replaced by its header and its rows
    decompressed: zlib's output is not the same on every build (CPython 3.14 on Windows
    compresses differently), so a recorded image is compared by its pixels."""

    def rows(part: Part) -> Part:
        if not isinstance(part, ImagePart):
            return part
        data, at, kept = part.data, 8, [part.data[:8]]
        idat = b""
        while at < len(data):
            length = int.from_bytes(data[at : at + 4], "big")
            kind = data[at + 4 : at + 8]
            if kind == b"IDAT":
                idat += data[at + 8 : at + 8 + length]
            else:
                kept.append(data[at : at + 12 + length])
            at += 12 + length
        return dataclasses.replace(part, data=b"".join(kept) + zlib.decompress(idat))

    state = State(parts=tuple(rows(part) for part in wire_request.state.parts))
    return dataclasses.replace(wire_request, state=state)


def test_the_minimum_image_side_is_what_the_wire_refused() -> None:
    """The probe, recorded 2026-10-22 outside the leaf: each size got a 400 exactly when a
    side was below the minimum. Fit refuses those same sizes before any call; a leaf called
    past Fit sends them and meets the wire's refusal."""
    directory = WITH_IMAGES.directory / SIDE_PROBE
    recorded = sorted(path.name for path in directory.iterdir())
    assert recorded == sorted(f"{w}x{h}.json" for w, h in IMAGE_SIDE_PROBES)
    for width, height in IMAGE_SIDE_PROBES:
        cassette = load_cassette(side_probe_path(width, height))
        assert _pixels(cassette.wire_request) == _pixels(side_probe(width, height))
        status = cassette.response["status"]
        below = min(width, height) < _MIN_IMAGE_SIDE
        assert status == (400 if below else 200), (width, height)
        wire = Wire(cassette.response["body"], status=status, headers=cassette.response["headers"])
        leaf = wire.leaf(**WITH_IMAGES.options())
        if below:
            error = _raised(leaf, cassette.wire_request)
            assert (error.status_code, error.is_retryable) == (400, False)
            assert len(wire.sent) == 1
            refused = Wire(cassette.response["body"], status=status)
            _, outcome = _ask(WITH_IMAGES, cassette.wire_request, refused)
            assert isinstance(outcome, DoesNotFit)
            assert outcome.check == "min_image_side"
            assert refused.sent == []
        else:
            response = _call(leaf, cassette.wire_request)
            assert isinstance(response.answers["noul_red"], NoulAnswer)
            assert body_key(wire.body_sent) == body_key(cassette.request["body"])


def test_the_estimate_covers_the_side_probes_usage() -> None:
    estimator = SystemOne(api_key="k").estimator
    for width, height in IMAGE_SIDE_PROBES:
        cassette = load_cassette(side_probe_path(width, height))
        if cassette.response["status"] != 200:
            continue
        request = cassette.wire_request
        estimate = estimator.estimate(request.state, request.questions)
        used = cassette.response["body"]["usage"]["input_tokens"]
        assert used <= estimate.tokens * (1 + estimate.margin), (width, height)


# ----------------------------------------------------------------------------------------
# Fit, against the record and the recorded requests
# ----------------------------------------------------------------------------------------


def _ask(
    configuration: Configuration, wire_request: WireRequest, wire: Wire | None = None
) -> tuple[Wire, object]:
    wire = wire if wire is not None else _answering({}, configuration.model)
    provider = Dispatcher(wire.leaf(**configuration.options()))
    request = Request(
        state=wire_request.state, questions=wire_request.questions, deadline=Deadline.after(5)
    )
    try:
        return wire, asyncio.run(provider.answer(request))
    except DoesNotFit as refused:
        return wire, refused


@pytest.mark.parametrize(
    ("configuration", "name", "check"),
    [
        *(pytest.param(c, FAILURE, "max_levels", id=f"{c.model}-levels") for c in CONFIGURATIONS),
        pytest.param(CONFIGURATIONS[0], "over-limit", "max_question_tokens", id="tokens"),
    ],
)
def test_preflight_refuses_what_the_wire_refused(
    configuration: Configuration, name: str, check: str
) -> None:
    wire_request = decode(load(configuration, name)["wire_request"])
    wire, refused = _ask(configuration, wire_request)
    assert isinstance(refused, DoesNotFit)
    assert refused.check == check
    assert wire.sent == []


@pytest.mark.parametrize(
    ("image", "check"),
    [
        pytest.param(ImagePart(data=png(_MIN_IMAGE_SIDE - 1, 320)), "min_image_side", id="small"),
        pytest.param(
            ImagePart(data=b"GIF89a", media_type="image/gif"), "image_media_types", id="gif"
        ),
    ],
)
def test_an_image_the_record_refuses_does_not_fit_before_any_call(
    image: ImagePart, check: str
) -> None:
    wire_request = _one(_NOUL, State(parts=(image,)))
    wire, refused = _ask(WITH_IMAGES, wire_request)
    assert isinstance(refused, DoesNotFit)
    assert refused.check == check
    assert wire.sent == []


@pytest.mark.parametrize(("configuration", "name", "asked"), _CASES)
def test_every_recorded_request_plans_one_wire_request_with_no_warning(
    configuration: Configuration, name: str, asked: WireRequest
) -> None:
    leaf = SystemOne(api_key="k", **configuration.options())
    request = Request(state=asked.state, questions=asked.questions, deadline=Deadline.after(5))
    plan = _preflight(request, leaf.capabilities, leaf.estimator, Fail())
    assert len(plan.wire_requests) == 1
    assert plan.outcome_warnings == ((),)
    assert plan.response_warnings == ()
