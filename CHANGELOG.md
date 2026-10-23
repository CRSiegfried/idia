# Changelog

All notable changes to idia are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release may break the API; each break is listed under **Changed** or **Removed**. A new question kind, a new elicitation or a new validation rule is a minor release and is listed here.

## [Unreleased]

## [0.1.0] - 2026-10-23

The first release.

### Added

- The core vocabulary: every shape from `State` to `Response`; `Capabilities` with `Rounding`, `UNBOUNDED`, the opt-in feature maps and the image fields; the closed kinds (`noul`, `choice`, `score`, `extraction`, `box`) and elicitations (`passthrough`, `logprobs`) with their candidates; the answer union with `Refusal`, `Fault` and `Rule`; `validate` and `validate_answers`; `noul_complement`; the `Leaf`, `Provider` and `TokenEstimator` protocols; and the errors under `IdiaError`, a `ProviderError`'s text filtered of key-shaped substrings.
- Fit: preflight of a request against a provider's capabilities, refusing with `DoesNotFit` before any call, `unchecked-limit` warnings for limits nobody knows, greedy packing of overflowing questions into wire requests that repeat the state, and `OverflowPolicy` with `Fail` as the default.
- Dispatch: `Dispatcher`, the one loop that runs a fit plan inside the caller's `Deadline`, with `RetryPolicy`, `ConcurrencyLimit` and `RateLimiter`, one `Outcome` per wire request with its attempts and provenance, the re-ask after a fault, and a stop after a credential error; `Dispatcher.plan` for the plan alone; `Router` and `Fallback` with their derived capabilities; `aclose()` and `async with` on every provider.
- `idia.provider(name, ...)` and the registry of wire leaves, each behind an optional extra whose name the import error gives; `idia.leaf(name, ...)`, which builds the same leaf unwrapped, so a caller can wrap a leaf of its own around it, such as one counting every call; and `idia.provider_names()`, the registered names:
  - `system-one` (extra `system-one`): System One's decision models, with images and, on the models that take it, `box`;
  - `openai-decisions` and `openai-chat` (extra `openai`): OpenAI's Decisions API, and a chat model read by the `logprobs` elicitation;
  - `cloudflare-clef` and `cloudflare-clairic` (extra `cloudflare`): Workers AI's Clef for the label kinds and Clairic for extraction, and `cloudflare()` in `idia.cloudflare`, a router over the two sharing one client.
- `HttpLeaf` in `idia.providers.http`, the base a leaf over a raw HTTP API may inherit: client lifecycle, one POST under the deadline, and the status table.
- Test doubles: `Scripted`, a leaf answered from a script, and `Recorded`, which records a leaf's calls under the hash of the canonical wire request and replays them in modes `none`, `once` and `new`.
- The conformance suite, `idia.conformance`: `leaf_checks` (L1 to L31) and `provider_checks` (P1 to P23), run with `run_checks` against a harness built from exchanges, so a leaf written outside idia can check itself.
- The spec in `docs/spec`.
- The public API: the names in `idia.__all__` and in the `__all__` of `idia.cloudflare`, `idia.conformance`, `idia.providers.http`, `idia.providers.system_one`, `idia.providers.openai_decisions`, `idia.providers.chat`, `idia.providers.cloudflare`, `idia.providers.scripted` and `idia.providers.recorded`; a public name is removed or changed only after a minor release of `DeprecationWarning`.

### Notes

- The binding to ommat ships with ommat, as `ommat[idia]`; idia has no dependency on ommat.

[Unreleased]: https://github.com/CRSiegfried/idia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CRSiegfried/idia/releases/tag/v0.1.0
