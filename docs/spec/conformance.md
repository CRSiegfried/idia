# The conformance suite

The checks every leaf and every provider passes before it is used, shipped as `idia.conformance` in the core package with no extra, so a leaf written outside idia can run them too.
The suite imports only idia and the standard library: no test runner, no HTTP client.

## Two halves

- **The leaf half**, `leaf_checks(make_harness)`, checks L1 to L31 against a leaf: it satisfies `Leaf` and is not a `Provider` until wrapped; its record is the declared, frozen `Capabilities`; one call sends one wire request; it raises `ProviderError` and nothing else, on the loop it was first used on and on a later one, a body it cannot read non-retryably, and ends by its deadline; every mapped answer passes `validate` with the declared rounding; a recorded success comes back with the same keys, no fault, and every answer as the wire gave it; a broken answer is a `Fault` whose siblings stand; it fills the provenance only it knows; an option it does not know gives exactly one more warning, `unknown-option` naming it, and changes neither the answers nor what is sent, the latter held by the harness serving `Success` only to the recorded request (L29); and after `aclose` a call raises a `ProviderError` that is not retryable and sends nothing (L30, not applicable to a leaf with no `aclose`; it runs on the leaf as the harness builds it, so a leaf whose client it creates tests that client's refusal in its own tests); and its image fields agree with its record, built with each opt-in feature on and off: with the image modality, one image carrier and byte caps that are not 0 (a cap of 0 the constructor already refuses); without it, no carrier and each field 0, or no media types, or unknown, or else the values an opt-in feature that adds images brings (L31).
  L8 has two halves, each on a fresh leaf: a success, which must send one wire request, and a retryable failure, a 429 or, for a harness that cannot make a status, a stall the leaf's deadline cuts, which must also send one; after a 429 the same leaf is called again over the success, which must send one more; it is not applicable only when neither half can be made, and a pass notes the half not run, a retryable half run over a stall rather than a 429, and a stall that did not end in a retryable `ProviderError`, which leaves the retryable half unexercised.
  Three checks read source instead of running it: the leaf's class, and each of its bases defined in the same module, declares no capabilities field, in any case, nor `capabilities` itself, as a class attribute, so a wire model's fields beside it are not read (L4); the module imports nothing of the call loop and keeps no `ProviderError` from it: an `except` that names it or a subclass is allowed only when its body ends in a bare `raise`, returns nowhere (functions defined inside it aside), and no loop in its function encloses it, as a recorder that stores the error and lets it go on does, and a bare `except`, `except BaseException` or `except Exception` is allowed only when its body ends by raising, its last statement a `raise` (L10); it computes no `1 - p` and refers to no `noul_complement`, by any name (L21), outside an allowlist for `1 - p` that holds only the OpenAI chat leaf's off-label mass, one less the summed label masses, so every `1 - x` in the leaf's module is flagged whatever `x` is, and arithmetic of that form that is no complement belongs in another module, since only the leaf's own is read.
  They read only the module that defines the leaf's class, and say so.
- **The provider half**, `provider_checks(make_provider)`, checks P1 to P23 against a leaf wrapped in a `Dispatcher` or a composite: preflight refuses before any wire request and warns an unknown limit; one outcome per wire request in plan order, with its provenance and no raw body; a retryable failure and an answer give one outcome with one failed attempt; a partial failure returns a `Response`, a total one raises `RequestFailed`; a fault is never retried; every answer passes `validate` exactly once, and nothing in the provider runs it again.
  P20 serves a 401 and P21 a 403 on every call with the question cap at one, so a member given two or more questions has several wire requests: each member makes exactly its first call, its first outcome lists that one failed attempt with the status, not retryable, and every later outcome of that member is left unsent with the same error object; a router whose members each hold one wire request has nothing to leave unsent and passes with a note.
  P22 gives a request that fits a deadline already passed: no member is called, nothing is sent, `RequestFailed` is raised, and every outcome is a `DeadlineExceeded` with no attempts.
  A verdict covers only the provider under test: that idia's own core imports nothing from `providers/` and that the loop's modules import neither validation function are idia's own tests, not checks of the suite.
  P14 to P16 run for composites only, P17 for a router, P18 for a fallback; P20 to P23 run for every shape.
  P23, for a provider whose record takes images, adds to the request's state, for each image field the record knows, an image that breaks that field and fits the rest: one over `max_image_bytes` and two over `max_total_image_bytes`, with the caps set small through `overrides`; one a pixel under `min_image_side`; a GIF declared `image/gif` and an untyped WebP, each where the set lacks it.
  The images are built as PNG, or as JPEG where the set takes JPEG and not PNG; a record taking neither runs the media-type drivers alone.
  Each is refused with `DoesNotFit` naming the field it breaks, nothing sent and no member called; for a router, whose members each see the whole state, a member that takes no images may refuse it on `modalities` first, which passes too.
  With no image modality or no known image field, P23 is not applicable.
  P18 runs twice, with the first member failing every call with a failure that is not a credential error and then with a 401: each time the next member is passed the same request with the request's own deadline, so it gets what remains of it, not a fresh one, and its first outcome lists the failed member's attempts.
  P19 runs for a wrapped leaf only: in a temporary directory it records the leaf under the recorded leaf in mode `once`, behind a `Dispatcher` with `RetryPolicy.without_delay()`, for an answer, a 503 then an answer (whose recording must show a failed attempt), and a failure; then replays each log in mode `none` over a fresh leaf, and asserts that no wire request was sent and that the `Response` or `RequestFailed` is the recorded one, latency and start times aside, errors compared by their fields.
  A harness that cannot make one of the three runs the others and passes with a note naming the log it skipped, as a leaf with no statuses skips the 503; with none of them it is not applicable, as for a composite.

Each half takes a zero-argument factory, and each check builds a fresh harness from it, so checks share no state and run in any order.
Each `Check` carries its `id` and a one-line `description`, so the checks `leaf_checks` and `provider_checks` return are the full table of IDs.

## The harness

A harness is what the leaf's or provider's own tests supply.
It speaks to the suite in **exchanges**, what the wire does on one call, and turns each into its own transport's terms:

| Exchange | The wire |
|---|---|
| `Success(kind=None)` | The recorded success for `wire_request(kind)`, served only to the request recorded for it: a harness fails a call whose request differs; with no kind, the one answering `request()` |
| `Failure(status=None, retry_after=None, request_id=None)` | A failure the wire or the runtime reports: the recorded one, or one of that status with those headers; `Failure()` with no status is a failure that is not retryable and is not a credential error (401 or 403); on a wire API it carries a status, on a runtime with no statuses none, and a harness with no statuses raises `CheckNotApplicable` for a `Failure` with a status |
| `Unreadable(which)` | A body that is not JSON, fails the top-level schema, or holds no answers map |
| `Perturbed(kind, how)` | The success with one sum moved outside tolerance, one key removed and one added, or a `noul` outside [0, 1] |
| `Error(which)` | A connection error or a timeout |
| `Stall()` | No answer at all, from the wire or the runtime |
| `RefusalOnWire(kind)`, `CandidateShape(kind)` | The recorded refusal, or a candidate shape for an admitted kind |

A `LeafHarness` has `capabilities` (the declared record), `prompt_hidden` (true when the wire API hides the prompt, so L26 accepts a response whose `prompt_sent` is `None`), `make(exchanges, features=None)` returning a fresh leaf whose wire plays the exchanges in order, `capabilities_for(features)`, `wire_request(kind)` and `expected(kind)` (the answers as mapped), `requests_sent()`, `sdk_retries()` and `client_closed(*, injected)`.
With no `features`, `make` builds the leaf with its record's `enabled_features`, so every check sees the leaf as declared; a leaf with several configurations runs the suite once per configuration.
`capabilities_for(features)` is the record the leaf must declare with those features on, and L7 compares the built record against it with each opt-in feature on and off, so the fields a feature changes are checked, and checks that `feature_kinds` and `feature_modalities` do not change.

An HTTP harness sends `Failure.retry_after` as `retry-after-ms` beside a different, conflicting `retry-after`, so L12 also checks that the millisecond header wins.
A leaf on a raw HTTP client reports its client's transport-level retries from `sdk_retries()`: they count as SDK retries for L9, which then runs rather than being not applicable.

A `ProviderHarness` has `shape` (`"leaf"`, `"router"` or `"fallback"`), `members`, `expected_capabilities` (for a composite, the record it should derive, from members that include a known and an unknown token limit, one truncating silently), `make(exchanges, *, overrides=None)` with one exchange sequence per member and `overrides` applied to every member leaf's record, `leaf(exchanges)`, the wrapped leaf alone and unwrapped, whose wire plays the exchanges (a composite raises `CheckNotApplicable`), `request()`, `requests_sent()`, `member_calls()`, each member's received wire requests with their deadlines since the last `make`, and `member_names()`, each member leaf's name and model.
Members are in a router's mapping order or a fallback's sequence order.
Each dispatcher it builds has a zero-delay retry policy.

## Verdicts

A check passes, fails with an `AssertionError`, or raises `CheckNotApplicable(reason)` when the harness cannot drive it: a leaf with no SDK has no SDK retries.
A harness's `make` raises it for an exchange its transport cannot make.
Where a check has several drivers and the harness can make only some, it runs those and passes with a note naming what was not run: P12, for one, runs whichever of its moved-sum and unreadable-body drivers the harness can make, the moved-sum driver only where the request holds a question whose sum can move; L8 and P19 do the same.
L26 passes a wire response with no `request_id`, or with `usage.output_tokens` `None`, with a note naming what was missing, since a model run in process has no wire to assign an id and writes no output tokens; a leaf never invents either, and the input token count, the runtime's own count of what it encoded where there is no wire, is still required.
P8 passes an outcome with no `request_id` the same way, with a note.
L25, where no expected extraction has a null `value` and a `guess`, checks only that each extraction answer's `value` and `guess` are those expected, so the leaf fills no guess, and notes that it did not check a guess being kept as given and not moved into `value`; with no expected extraction answer at all it is not applicable.

`await run_checks(checks)` runs the checks one at a time and returns one `Verdict(id, result, reason)` each, `result` being `"pass"`, `"fail"` or `"not_applicable"`; a `CheckNotApplicable` gives `not_applicable`; any other `Exception` is a failure; a `BaseException` that is not an `Exception`, such as cancellation, propagates.
The checks must not run concurrently: the `validate` spy patches every module's binding of `validate` while a check runs.

## The scripted leaf under the suite

The scripted leaf passes both halves, wrapped for the provider half.
A script cannot stand in for an SDK, a status mapper, a body or an HTTP client, and the scripted leaf has no `aclose`, so L9, L12, L13, L15, L24 and L30 are not applicable to it, and idia's tests assert that set exactly, both for the scripted leaf and for it under the recorded leaf while recording.
In the provider half, the scripted leaf's record takes no images, so a wrapped leaf is not applicable to P14 to P18 and P23, a router to P6, P18, P19 and P23, and a fallback to P6, P17, P19 and P23, and the tests assert each set exactly.
A scripted refusal is a `Refusal` answer, so L23 runs for a record whose `returns_refusal` holds.

## Running it under pytest

A leaf's tests parametrize one test over the checks, each case named by its ID, and run each with `asyncio.run(check.run())`, turning `CheckNotApplicable` into a skip.
idia's own `tests/_conformance.py` holds that helper.
Outside pytest, `asyncio.run(run_checks(leaf_checks(make_harness)))` returns one `Verdict` per check.
idia's own harnesses are test code and are not installed, so a leaf's author writes the `LeafHarness` over the leaf's own transport.
