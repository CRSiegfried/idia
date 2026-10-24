"""The one run against the real GLiNER2 model, which records the stored outputs the other
tests replay. Skipped unless ``IDIA_LOCAL_MODEL=1``, which also sets ``HF_HUB_OFFLINE=1``:
the weights at the pinned revision must already be cached, and nothing is downloaded.

It loads the model through the leaf's own loader, runs each stored case again, checks that
the spans are the same and each score within :data:`TOLERANCE` of the stored one, and writes
the outputs back with the runtime's versions; then it asks the loaded leaf through a
dispatcher. Run it once per model: ``IDIA_LOCAL_MODEL=1 uv run pytest -m local_model -s``.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import importlib.metadata
import platform
from typing import Any, Final

import pytest

from idia import Deadline, Dispatcher, ExtractionAnswer, Request
from idia.providers._gliner import _MODEL, _REVISION, _load_model
from tests._gliner_cases import (
    ASKED,
    SENDER,
    STATE,
    STORED,
    TOPICS,
    load_outputs,
    record_outputs,
)

pytestmark = [
    pytest.mark.local_model,
    # Loading the checkpoint, torch warns that torch.jit.script is deprecated.
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]

TOLERANCE: Final = 0.01
"""How far a span's score may move between runtime versions before the run fails."""


def _span_keys(spans: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    return [(span["text"], span["start"], span["end"]) for span in spans]


def test_the_real_model_gives_the_stored_outputs_and_records_them() -> None:
    pytest.importorskip("gliner2")
    stored = load_outputs()
    runtime, commit = _load_model(_MODEL, _REVISION)
    assert commit == _REVISION
    for text, labels, threshold in ASKED:
        output = runtime.extract_entities(text, labels, threshold)
        then, then_tokens = stored[(text, labels, threshold)]
        assert runtime.input_tokens(text, labels) == then_tokens
        assert output["entities"].keys() == then["entities"].keys()
        for label, spans in output["entities"].items():
            assert _span_keys(spans) == _span_keys(then["entities"][label]), label
            for new, was in zip(spans, then["entities"][label], strict=True):
                assert abs(new["confidence"] - was["confidence"]) <= TOLERANCE, label
    versions = {
        package: importlib.metadata.version(package)
        for package in ("gliner2", "torch", "transformers", "huggingface_hub")
    }
    record_outputs(
        runtime,
        ASKED,
        STORED,
        {"revision": commit, "python": platform.python_version(), **versions},
    )


def test_the_loaded_leaf_answers_through_a_dispatcher() -> None:
    gliner = pytest.importorskip("idia.providers.gliner")

    async def ask() -> None:
        leaf = gliner.Gliner()
        await leaf.load()
        async with Dispatcher(leaf) as provider:
            request = Request(state=STATE, questions=(SENDER, TOPICS), deadline=Deadline.after(60))
            response = await provider.answer(request)
        sender = response.answers["sender"]
        assert isinstance(sender, ExtractionAnswer)
        assert sender.value == "Dana Whitfield"
        topics = response.answers["topics"]
        assert isinstance(topics, ExtractionAnswer)
        assert topics.value == ("Q3 budget", "hiring")
        assert topics.confidence is None
        outcome = response.outcomes[0]
        assert outcome.snapshot == _REVISION
        assert outcome.request_id is None
        assert outcome.usage is not None
        assert (outcome.usage.input_tokens, outcome.usage.output_tokens) == (33, None)

    asyncio.run(ask())
