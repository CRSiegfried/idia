"""A minimal leaf the registry tests load by name."""

from __future__ import annotations

from idia import Deadline, WireRequest, WireResponse
from tests._support import FixedEstimator, capabilities


class FakeLeaf:
    def __init__(self, *, model: str = "fake-1") -> None:
        self.name = "fake"
        self.model = model
        self.capabilities = capabilities()
        self.estimator = FixedEstimator()

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        return WireResponse(answers={}, elicitation="passthrough", calibrated=True)


class CheckingLeaf(FakeLeaf):
    """A leaf whose constructor refuses a value of the wrong type with its own TypeError."""

    def __init__(self, *, model: str = "fake-1") -> None:
        if not isinstance(model, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"the model is a str, not {type(model).__name__}")
        super().__init__(model=model)


class NotALeaf:
    def __init__(self) -> None:
        self.name = "not-a-leaf"
