"""Leaves that break one rule of the leaf half each, and a provider that breaks one rule of
the provider half, for the suite's own negative tests.

This module imports nothing from the call loop and computes no complement, so the review
checks, which read it, pass for every leaf here: each fails only the check it targets.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from collections.abc import Mapping
from dataclasses import replace
from types import ModuleType

from idia.conformance._scripted import ScriptedHarness
from idia.core import (
    WARNING_NO_SNAPSHOT,
    Capabilities,
    Deadline,
    MappedAnswer,
    Notice,
    Provider,
    ProviderError,
    Request,
    RequestFailed,
    Response,
    TokenEstimate,
    TokenEstimator,
    Usage,
    WireRequest,
    WireResponse,
    unknown_options,
)
from idia.providers.scripted import Reply, ScriptFunction


def _rules_copy() -> ModuleType:
    """A second copy of the validation module: its own functions, the same rules."""
    import idia.answers

    spec = importlib.util.spec_from_file_location("rules_copy", idia.answers.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RULES = _rules_copy()


class _Flat:
    def estimate(self, state: object, questions: object) -> TokenEstimate:
        return TokenEstimate(tokens=1, margin=0.0)


class OwnRules:
    """A leaf that validates with its own copy of the rules instead of calling validate."""

    def __init__(self, capabilities: Capabilities, script: ScriptFunction) -> None:
        self.name = "own-rules"
        self.model = "own-rules-1"
        self.capabilities = capabilities
        self.estimator: TokenEstimator = _Flat()
        self._script = script
        self._n = 0

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        self._n += 1
        result = self._script(wire_request, deadline)
        assert inspect.isawaitable(result)
        try:
            item = await asyncio.wait_for(result, max(deadline.remaining(), 0.0))
        except TimeoutError:
            raise ProviderError("stalled", is_retryable=True) from None
        if isinstance(item, ProviderError):
            raise item
        assert isinstance(item, Reply)
        mapped: Mapping[str, MappedAnswer] = item.answers
        answers, warnings = _RULES.validate_answers(  # the copy's own validate
            wire_request.questions, mapped, self.capabilities.rounding
        )
        snapshot = self.model if self.capabilities.snapshot_pinnable else None
        if snapshot is None:
            warnings += (Notice(code=WARNING_NO_SNAPSHOT, message="no snapshot"),)
        return WireResponse(
            answers=answers,
            elicitation="passthrough",
            calibrated=self.capabilities.calibrated,
            model_reported=self.model,
            snapshot=snapshot,
            prompt_sent="the prompt",
            usage=Usage(input_tokens=1, output_tokens=1, raw_usage={"tokens": 2}),
            request_id=f"own-{self._n}",
            raw_body={"answers": "as recorded"},
            warnings=warnings + unknown_options(wire_request.options, frozenset()),
        )


class OwnRulesHarness(ScriptedHarness):
    """The scripted wire behind a leaf that never calls validate."""

    def _build(self, capabilities: Capabilities, script: ScriptFunction) -> OwnRules:
        return OwnRules(capabilities, script)


class AsksAgain:
    """A provider that, when no wire request answered, asks once more before raising: past a
    credential error it sends again, as if the stop did not hold."""

    def __init__(self, inner: Provider) -> None:
        self._inner = inner

    @property
    def capabilities(self) -> Capabilities:
        return self._inner.capabilities

    async def answer(self, request: Request) -> Response:
        try:
            return await self._inner.answer(request)
        except RequestFailed:
            return await self._inner.answer(request)


class InventsAnUnsentOutcome:
    """A provider that, when no wire request answered, rewrites its last outcome as left
    unsent by a credential error: no attempts, no request id, and a 401 of its own that no
    earlier outcome holds."""

    def __init__(self, inner: Provider) -> None:
        self._inner = inner

    @property
    def capabilities(self) -> Capabilities:
        return self._inner.capabilities

    async def answer(self, request: Request) -> Response:
        try:
            return await self._inner.answer(request)
        except RequestFailed as failed:
            *earlier, last = failed.response.outcomes
            invented = replace(
                last,
                error=ProviderError("credentials refused", status_code=401),
                failed_attempts=(),
                request_id=None,
            )
            raise RequestFailed(replace(failed.response, outcomes=(*earlier, invented))) from None
