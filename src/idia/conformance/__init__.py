"""The conformance suite every leaf and every provider passes before it is used.

It has two halves. The leaf half, :func:`leaf_checks`, runs against a leaf a
:class:`LeafHarness` builds; the provider half, :func:`provider_checks`, against a wrapped
leaf or a composite a :class:`ProviderHarness` builds. A harness is what a provider's
author supplies: it turns each :class:`Exchange` into its own transport's terms. Each
:class:`Check` passes, fails with an ``AssertionError``, or raises
:class:`CheckNotApplicable`; :func:`run_checks` turns them into :class:`Verdict` values
without a test runner. The suite imports only idia and the standard library.
"""

from idia.conformance._base import (
    CandidateShape,
    Check,
    CheckNotApplicable,
    Error,
    Exchange,
    Failure,
    LeafHarness,
    Perturbed,
    ProviderHarness,
    RefusalOnWire,
    Stall,
    Success,
    Unreadable,
    Verdict,
    run_checks,
)
from idia.conformance._leaf import leaf_checks
from idia.conformance._provider import provider_checks

__all__ = [
    "CandidateShape",
    "Check",
    "CheckNotApplicable",
    "Error",
    "Exchange",
    "Failure",
    "LeafHarness",
    "Perturbed",
    "ProviderHarness",
    "RefusalOnWire",
    "Stall",
    "Success",
    "Unreadable",
    "Verdict",
    "leaf_checks",
    "provider_checks",
    "run_checks",
]
