# Changelog

All notable changes to idia are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release may break the API; each break is listed under **Changed** or **Removed**. A new question kind, a new elicitation or a new validation rule is a minor release and is listed here.

## [Unreleased]

### Changed

- The package's classifiers name the operating systems its tests run on, macOS and Linux, in place of `OS Independent`; Windows is not yet tested.

## [0.2.0] - 2026-11-15

### Added

- `extras` on `WireResponse` and `Outcome`, `Mapping[str, JsonValue]`, empty by default: the figures a wire reports beyond the shared vocabulary, under keys each leaf documents in the providers spec. It is no provenance detail and needs no reason. It is held read-only at every depth in the order built, and either constructor refuses a non-mapping with `TypeError`, and a key that is not a string, a value outside the JSON domain or a non-finite float with `ValueError`. Dispatch carries an answered outcome's `extras` from its wire response unchanged; a failed outcome has none. Clairic fills `{"passes": n}` from its usage; every other leaf gives it empty. The scripted leaf's `Reply` takes `extras`, the recorded leaf's log keeps it and leaves it out when empty, and conformance check P8 holds a wrapped leaf's outcome to the leaf's `extras` and a failed outcome to none.

### Changed

- The recorded leaf's log is format 4, and a format-3 log is refused until upgraded. `upgrade_log` brings a format-3 log to format 4, moving each logged pass count from the usage to `extras` and dropping `usage.passes` from the reasons and claims, with its keys and everything else unchanged; it still re-keys a format-2 log, now to format 4, and its `requests` default to none. A log that is not JSON is refused as not a recorded leaf's log.

### Removed

- `Usage.passes`, and `usage.passes` from `DETAILS` and `ProvenanceDetail`: a figure only one provider reports is no longer a typed field of the shared vocabulary. Read Clairic's passes from `outcome.extras["passes"]`. No leaf gives `usage.passes` a reason any longer, and `upgrade_log` no longer gives it `not-exposed`.

## [0.1.0] - 2026-10-23

The first release.

### Added

- The vocabulary: every shape from `State` to `Response`; the closed kinds (`noul`, `choice`, `score`, `extraction`, `box`) and elicitations (`passthrough`, `logprobs`) with their candidates; the answer union with `Refusal`, `Fault` and `Rule`; `noul_complement`; and the `Leaf`, `Provider` and `TokenEstimator` protocols.
- A question's wording: its instruction a string or a JSON object or array, and a `noul` optionally carrying `yes` and `no` labels that describe what counts as each. A leaf sends the wording as its wire takes it or refuses it, and never drops it: System One and Clef send both, and the chat leaf writes a noul's descriptions into its prompt.
- Key order kept: a JSON object's keys are held, sent, recorded and keyed in the order the caller built them, at every depth, in a JSON part, a structured instruction and the options, since key order is part of what a model reads and moves its answers. Each such value is held read-only, objects as read-only mappings and arrays as tuples, so a later change to the caller's object reaches nothing; a shape holding one cannot be pickled or deep-copied; and a value that contains itself or nests too deep is refused with `ValueError` naming the field when it is built. `==` on the shapes compares mappings without regard to order, so two equal shapes may be two different prompts, and `hash()` agrees with it.
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
- Test doubles: `Scripted`, a leaf answered from a script, and `Recorded`, which records a leaf's calls under the hash of the canonical wire request, key order included, in a format-3 log that keeps each key's request beside its calls, and replays them in modes `none`, `once` and `new`, refusing with `ValueError` a log of another format or one it cannot read and keeping a log's permission bits when it rewrites it; `upgrade_log(log, *, requests, reasons=None)`, which brings a format-2 log, written before this release, to format 3 offline, re-keying only the given requests whose key order cannot have differed, leaving out and counting every other entry, and filling each response's provenance reasons. The scripted leaf's `prompt_sent` is its wire request as compact JSON in the order built.
- The conformance suite, `idia.conformance`: `leaf_checks` (L1 to L32, L32 holding a leaf to a caller's key order in the body sent and its `prompt_sent`) and `provider_checks` (P1 to P23), run with `run_checks` against a harness built from exchanges, `AnyBody` among them, and an optional `body_sent()` on a leaf harness, so a leaf written outside idia can check itself.
- The spec in `docs/spec`.
- The public API: the names in `idia.__all__` and in the `__all__` of `idia.cloudflare`, `idia.conformance`, `idia.providers.http`, `idia.providers.system_one`, `idia.providers.openai_decisions`, `idia.providers.chat`, `idia.providers.cloudflare`, `idia.providers.scripted` and `idia.providers.recorded`; a public name is removed or changed only after a minor release of `DeprecationWarning`.

### Notes

- The binding to ommat ships with ommat, as `ommat[idia]`; idia has no dependency on ommat.

[Unreleased]: https://github.com/CRSiegfried/idia/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/CRSiegfried/idia/releases/tag/v0.2.0
[0.1.0]: https://github.com/CRSiegfried/idia/releases/tag/v0.1.0
