"""The GLiNER2 leaf's cases and its stored outputs: what the runtime returned for each input,
recorded by the gated run against the real model, and the model a test injects to play
them back.

The stored outputs are one JSON file, ``{"recorded_with": {...}, "cases": [...]}``: the
runtime's versions and the revision, then per call the text, the labels and the threshold
the leaf passed, the raw output of ``extract_entities`` and the input token count.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from idia import Question, State, TextPart
from idia.conformance import CheckNotApplicable, Exchange, Failure, Stall, Success
from idia.providers._gliner import GlinerRuntime

STORED: Final = (
    Path(__file__).parent / "stored" / "gliner" / "fastino" / "gliner2.5-small-v1" / "outputs.json"
)
THRESHOLD: Final = 0.5

TEXT: Final = (
    "Move the emails from Dana Whitfield about the Q3 budget and hiring to the Finance folder."
)
STATE: Final = State(parts=(TextPart(text=TEXT),))
SENDER: Final = Question(
    key="sender", kind="extraction", instruction="the sender", value_type="text"
)
TOPICS: Final = Question(
    key="topics", kind="extraction", instruction="the topics", value_type="text", is_list=True
)

Asked = tuple[str, tuple[str, ...], float]
"""Every argument the leaf passes the runtime: the text, the labels and the threshold."""

ASKED: Final[tuple[Asked, ...]] = (
    (TEXT, ("the sender", "the topics"), THRESHOLD),
    (TEXT, ("the sender",), THRESHOLD),
    (TEXT, ("the topics",), THRESHOLD),
)
"""The request the leaf half sends, and each one-question split the provider half sends."""

Stored = dict[Asked, tuple[Mapping[str, Any], int]]
"""For each call: the raw output of ``extract_entities`` and the input token count."""


def record_outputs(
    runtime: GlinerRuntime,
    asked: Iterable[Asked],
    path: Path,
    recorded_with: Mapping[str, str | None],
) -> None:
    """Run each case through ``runtime`` and write the outputs, with the versions, to
    ``path``."""
    cases = [
        {
            "text": text,
            "labels": list(labels),
            "threshold": threshold,
            "output": runtime.extract_entities(text, labels, threshold),
            "input_tokens": runtime.input_tokens(text, labels),
        }
        for text, labels, threshold in asked
    ]
    document = {"recorded_with": dict(recorded_with), "cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_outputs(path: Path = STORED) -> Stored:
    document = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    return {
        (case["text"], tuple(case["labels"]), case["threshold"]): (
            case["output"],
            case["input_tokens"],
        )
        for case in cast("list[dict[str, Any]]", document["cases"])
    }


def recorded_with(path: Path = STORED) -> Mapping[str, str | None]:
    document = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    return cast("Mapping[str, str | None]", document["recorded_with"])


OUTPUTS: Final = load_outputs()


class Playback:
    """The model a test injects: it plays the exchanges, one per pass, each success from
    the stored outputs for the inputs the leaf passed."""

    def __init__(self, exchanges: Sequence[Exchange] = ()) -> None:
        for exchange in exchanges:
            if isinstance(exchange, Failure) and exchange.status is not None:
                raise CheckNotApplicable("the model has no statuses")
            if not isinstance(exchange, Success | Failure | Stall):
                raise CheckNotApplicable(f"a model in process cannot make {exchange}")
        self._exchanges = list(exchanges)
        self._tokens: int | None = None
        self.calls = 0

    def extract_entities(
        self, text: str, labels: Sequence[str], threshold: float
    ) -> Mapping[str, Any]:
        self.calls += 1
        if not self._exchanges:
            raise RuntimeError(f"the exchanges ran out at call {self.calls}")
        exchange = self._exchanges.pop(0)
        if isinstance(exchange, Failure):
            raise RuntimeError("the model failed")
        if not isinstance(exchange, Success):
            time.sleep(1.0)
            raise RuntimeError("stalled: no answer")
        asked = (text, tuple(labels), threshold)
        stored = OUTPUTS.get(asked)
        if stored is None or (exchange.kind is not None and asked != ASKED[0]):
            raise RuntimeError(f"no stored output for {asked!r}")
        output, self._tokens = stored
        return output

    def input_tokens(self, text: str, labels: Sequence[str]) -> int:
        if self._tokens is None:
            raise RuntimeError("no pass ran")
        return self._tokens


class Answering(Playback):
    """A model that answers every pass with ``output`` and counts ``tokens``."""

    def __init__(self, output: Mapping[str, Any], tokens: int = 1) -> None:
        super().__init__()
        self._output = output
        self._count = tokens

    def extract_entities(
        self, text: str, labels: Sequence[str], threshold: float
    ) -> Mapping[str, Any]:
        self.calls += 1
        return self._output

    def input_tokens(self, text: str, labels: Sequence[str]) -> int:
        return self._count
