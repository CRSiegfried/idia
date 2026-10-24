"""The GLiNER2 leaf's cases and its stored outputs: what the runtime returned for each input,
recorded by the gated run against the real model, and the model a test injects to play
them back.

The stored outputs are one JSON file, ``{"recorded_with": {...}, "cases": [...]}``: the
runtime's versions and the revision, then per call the text, the labels and the threshold
the leaf passed, the raw output of ``extract_entities`` and the input token count.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from idia import Question, State, TextPart
from idia.conformance import (
    CheckNotApplicable,
    Exchange,
    Failure,
    Perturbed,
    Stall,
    Success,
    Unreadable,
)

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
NOTHING: Final = "the phone number"
"""A label that matches nothing in :data:`TEXT`."""

NON_ASCII: Final = "Überweisen Sie 300 € an Zoë Müller in Zürich, Straße des 17. Juni."
"""A text whose spans start past characters of more than one byte, so that a character
offset and a byte offset differ."""
NON_ASCII_LABELS: Final = ("the recipient", "the city", "the street")

Asked = tuple[str, tuple[str, ...], float]
"""Every argument the leaf passes the runtime: the text, the labels and the threshold."""

RECORDED: Final[Mapping[Asked, Mapping[str, tuple[str, ...]]]] = {
    (TEXT, ("the sender", "the topics"), THRESHOLD): {
        "the sender": ("Dana Whitfield",),
        "the topics": ("Q3 budget", "hiring"),
    },
    (TEXT, ("the sender",), THRESHOLD): {"the sender": ("Dana Whitfield",)},
    (TEXT, ("the topics",), THRESHOLD): {"the topics": ("Q3 budget", "hiring")},
    (TEXT, ("the sender", NOTHING), THRESHOLD): {
        "the sender": ("Dana Whitfield",),
        NOTHING: (),
    },
    (NON_ASCII, NON_ASCII_LABELS, THRESHOLD): {
        "the recipient": ("Zoë Müller",),
        "the city": ("Zürich",),
        "the street": ("Straße des 17. Juni",),
    },
}
"""Each stored call, with the spans' texts the real model gives for each label."""

ASKED: Final[tuple[Asked, ...]] = tuple(RECORDED)
"""The request the leaf half sends, each one-question split the provider half sends, a
label that finds nothing, and a text that is not ASCII."""

ESTIMATED: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "Please forward the invoice from Northwind Traders dated March 3 to Priya Raman in "
        "accounts payable before Friday.",
        ("the sender", "the recipient", "the date", "the company"),
    ),
    (
        "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, "
        "tertiam qui ipsorum lingua Celtae, nostra Galli appellantur.",
        ("the people", "the place", "the number", "the language"),
    ),
    (NON_ASCII, NON_ASCII_LABELS),
)
"""An English, a Latin and a non-ASCII text, each with several labels: the estimate with
its margin must cover the tokenizer's count of each. The last is stored with its count."""

Stored = dict[Asked, tuple[Mapping[str, Any], int]]
"""For each call: the raw output of ``extract_entities`` and the input token count."""


def record_outputs(
    cases: Sequence[Mapping[str, Any]], path: Path, recorded_with: Mapping[str, str | None]
) -> None:
    """Write ``cases``, each the text, labels, threshold, output and input token count of
    one call, with the versions, to ``path``: whole or not at all, through a file beside it
    replaced in one step."""
    document = {"recorded_with": dict(recorded_with), "cases": list(cases)}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with open(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


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

_NO_EXTRA_KEY: Final = (
    "the runtime answers by label and the leaf maps by question key, so no key without a "
    "question can reach validation"
)

_UNREADABLE: Final[Mapping[str, object]] = {
    "not_json": "not a mapping",
    "schema": {"spans": []},
    "no_answers": {"entities": []},
}
"""What the model returns for each unreadable body: no mapping, and two with no entities map."""


class Playback:
    """The model a test injects: it plays the exchanges, one per pass, each success from
    the stored outputs for the inputs the leaf passed."""

    def __init__(self, exchanges: Sequence[Exchange] = ()) -> None:
        for exchange in exchanges:
            if isinstance(exchange, Failure) and exchange.status is not None:
                raise CheckNotApplicable("the model has no statuses")
            if isinstance(exchange, Perturbed) and exchange.how == "keys":
                raise CheckNotApplicable(_NO_EXTRA_KEY)
            if not isinstance(exchange, Success | Failure | Stall | Unreadable):
                raise CheckNotApplicable(f"a model in process cannot make {exchange}")
        self._exchanges = list(exchanges)
        self._tokens: int | None = None
        self.calls: int = 0

    def extract_entities(
        self, text: str, labels: Sequence[str], threshold: float
    ) -> Mapping[str, Any]:
        self.calls += 1
        if not self._exchanges:
            raise RuntimeError(f"the exchanges ran out at call {self.calls}")
        exchange = self._exchanges.pop(0)
        if isinstance(exchange, Failure):
            raise RuntimeError("the model failed")
        if isinstance(exchange, Unreadable):
            self._tokens = 1
            return cast("Mapping[str, Any]", _UNREADABLE[exchange.which])
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
