# Changelog

All notable changes to idia are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release may break the API; each break is listed under **Changed** or **Removed**. A new question kind, a new elicitation or a new validation rule is a minor release and is listed here.

## [Unreleased]

A JSON value's key order is part of what a model reads, and changes its answers on some requests, so idia now keeps the order a caller builds everywhere: in what is sent, in what is recorded, and in the key a recording is filed under.

### Changed

- A `JsonPart`'s value, a structured `Question.instruction` and each value of `Request.options` and `WireRequest.options` are held read-only all the way down, in the order given: each object a read-only mapping keeping its key order, each list a tuple. A later change to the caller's object no longer reaches a request. A list inside a JSON part or an option now reads back as a tuple; the bytes a leaf sends do not change.
- A value that contains itself, or is nested too deep, in a JSON part, a structured instruction or an option is refused when it is built, with `ValueError` naming the field (`... contains itself`, `... nests too deep`). Before, an over-deep JSON part reached the leaf and was refused as a `ProviderError`.
- `JsonPart.value` is left out of `hash()`, as a structured instruction and the options already were, so a `State` holding a JSON object can be hashed. `==` still compares mappings without regard to key order: two equal shapes may be two different prompts.
- The recorded leaf's key keeps key order, so two wire requests that differ only in a mapping's key order are recorded and replayed apart. Every recorded key changes.
- Recorded logs are format 3: each key holds the request it was computed from beside its calls, and everything is written in the order held, but for the keys of `calls` and any set, which are sorted. A `Recorded` built on a log of any other format raises `ValueError` naming `upgrade_log`, as does a stored request that does not hash to its key. A log holds the caller's state as written; secret filtering covers headers and bodies, not a caller's content.
- `upgrade_log(log, *, requests, reasons=None)` brings a format-2 log to format 3. It re-keys a given request only when no mapping in it has two or more keys at any depth, so its order cannot have differed from the one recorded; it refuses with `ValueError`, writing nothing, a request with such a mapping whose old key is in the log, and a log already in format 3. It still fills each response's provenance reasons, and returns how many entries no request re-keyed, which are left out, so a replay of one is a `ReplayMiss`. A re-keyed response keeps the sorted `prompt_sent` and raw body its old log holds. Break: `requests` is required, and the return value counts the entries left out.
- The scripted leaf's `prompt_sent` is its wire request as compact JSON in the order built, no longer sorted.
- Conformance check P19 compares a replay's outcomes with the recording's in a rendering that keeps key order.

### Added

- Conformance check L32: each multi-key JSON part and structured instruction a leaf's record carries reaches the body sent and its `prompt_sent` in the order given, at every depth.
- `AnyBody(kind)` in `idia.conformance`, the recorded success served whatever body the leaf sends, and an optional `body_sent()` on a leaf harness; a harness without it runs L32 with the body half noted as not run.

### Fixed

- The OpenAI leaves' estimator measures a read-only mapping in a JSON part or a structured instruction as the object it holds, not as its `repr`.

## [0.1.0] - 2026-10-23

The first release.

### Added

- The vocabulary: every shape from `State` to `Response`; the closed kinds (`noul`, `choice`, `score`, `extraction`, `box`) and elicitations (`passthrough`, `logprobs`) with their candidates; the answer union with `Refusal`, `Fault` and `Rule`; `noul_complement`; and the `Leaf`, `Provider` and `TokenEstimator` protocols.
- A question's wording: its instruction a string or a JSON object or array, and a `noul` optionally carrying `yes` and `no` labels that describe what counts as each. A leaf sends the wording as its wire takes it or refuses it, and never drops it: System One and Clef send both, and the chat leaf writes a noul's descriptions into its prompt.
- `Capabilities`, every field stated: the kinds, modalities and limits, `Rounding`, `UNBOUNDED`, the opt-in feature maps, the image fields, and `carries_noul_descriptions` and `carries_structured_instructions`, which say what wording a leaf sends.
- Answer validation, `validate` and `validate_answers`: fixed rules per kind, with tolerances from the declared rounding, for the sum and for a chosen label, which may lie one rounding step below the highest probability, since a provider may round after it chooses; such a label is kept as given, with a `chosen-below-max` warning (`WARNING_CHOSEN_BELOW_MAX`) naming the question.
- The errors under `IdiaError`: `DoesNotFit` before any call, `RequestFailed` carrying the response, and `ProviderError`, the one error a leaf raises, its text filtered of key-shaped substrings.
- Fit: preflight of a request against a provider's capabilities, refusing with `DoesNotFit` before any call a request it cannot split to fit or a wording form the record does not carry, `unchecked-limit` warnings for limits nobody knows, greedy packing of overflowing questions into wire requests that repeat the state, and `OverflowPolicy` with `Fail` as the default.
- Dispatch: `Dispatcher`, the one loop that runs a fit plan inside the caller's `Deadline`, with `RetryPolicy`, `ConcurrencyLimit` and `RateLimiter`, one `Outcome` per wire request with its attempts and provenance, the re-ask after a fault, and a stop after a credential error; `Dispatcher.plan` for the plan alone; `Router` and `Fallback` with their derived capabilities; `aclose()` and `async with` on every provider.
- Provenance on every outcome to one profile: each of `DETAILS`, from the model reported to the request id, a value or an `UnreportedReason`, the codes `REASONS` lists, in `unreported` (`not-reported`, `not-pinned`, `not-sent`, `hidden`, `not-applicable`, `not-exposed`), the details a source only `claimed`, the runtime's `versions` and the client libraries' `client_versions`, idia's own among them, and each outcome's `rounding`; the conformance suite holds every leaf to it.
- `idia.provider(name, ...)` and the registry of leaves, each behind an optional extra whose name the import error gives; `idia.leaf(name, ...)`, which builds the same leaf unwrapped, so a caller can wrap a leaf of its own around it, such as one counting every call; and `idia.provider_names()`, the registered names:
  - `system-one` (extra `system-one`): System One's decision models, `jev-1.14.0` by default, with images and, on the models that take it, `box`; `jev-latest`, a moving alias, is offered without `box`;
  - `openai-decisions` and `openai-chat` (extra `openai`): OpenAI's Decisions API, and a chat model read by the `logprobs` elicitation;
  - `cloudflare-clef` and `cloudflare-clairic` (extra `cloudflare`): Workers AI's Clef for the label kinds and Clairic for extraction, and `cloudflare()` in `idia.cloudflare`, a router over the two sharing one client.
  - `gliner` (extra `gliner`): the GLiNER2 span extractor `fastino/gliner2.5-small-v1`, run in this process, for `extraction` of text from one text part, a value as the top-scoring span with its score as `confidence`, or a list. Its weights are pinned to a revision by default and loaded by an explicit `await leaf.load()`; a model passed in as `runtime` is reported at the `revision` the caller passes, and with none at no snapshot, with the `no-snapshot` warning; its usage counts input tokens with the model's own tokenizer. The leaf, `Gliner`, is in `idia.providers.gliner`, which is not yet part of the public API.
- `HttpLeaf` in `idia.providers.http`, the base a leaf over a raw HTTP API may inherit: client lifecycle, one POST under the deadline, and the status table.
- Test doubles: `Scripted`, a leaf answered from a script, and `Recorded`, which records a leaf's calls under the hash of the canonical wire request and replays them in modes `none`, `once` and `new`, refusing with `ValueError` a log it cannot read and keeping a log's permission bits when it rewrites it; `upgrade_log`, which brings a log written before the provenance profile up to it, offline.
- The conformance suite, `idia.conformance`: `leaf_checks` (L1 to L31) and `provider_checks` (P1 to P23), run with `run_checks` against a harness built from exchanges, so a leaf written outside idia can check itself.
- The spec in `docs/spec`.
- The public API: the names in `idia.__all__` and in the `__all__` of `idia.cloudflare`, `idia.conformance`, `idia.providers.http`, `idia.providers.system_one`, `idia.providers.openai_decisions`, `idia.providers.chat`, `idia.providers.cloudflare`, `idia.providers.scripted` and `idia.providers.recorded`; a public name is removed or changed only after a minor release of `DeprecationWarning`.

### Notes

- The binding to ommat ships with ommat, as `ommat[idia]`; idia has no dependency on ommat.

[0.1.0]: https://github.com/CRSiegfried/idia/releases/tag/v0.1.0
