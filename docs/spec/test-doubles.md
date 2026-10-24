# Test doubles

The leaves idia's own tests, and a leaf's or composite's tests, answer from instead of a wire API.
They ship in the core package with no extra and are not in the provider registry: a test builds one and wraps it itself.

## The scripted leaf

`Scripted(*, capabilities, script, name="scripted", model="scripted-1", estimator=None)`, in `idia.providers.scripted`, satisfies `Leaf` by its shape, with no base class.
It is not a `Provider` until wrapped.
Its name, model and capabilities record are the test's; with no estimator it counts one token per four characters of the rendered wire request.

- **The script** is a sequence of items consumed one per call, or a function `(wire_request, deadline) -> item`, sync or async, called once per call.
  An async function is awaited under the call's deadline; one that stalls past it ends in a retryable `ProviderError`, as a wire leaf's timeout does, so an async script is how a test simulates a stall.
  A sync function is not timed and must not sleep.
  An item is a `Reply(*, answers, model_reported=None, pinned=True, usage=None, warnings=(), versions={}, client_versions={}, unreported=None, claimed=frozenset())`, a bare mapping of question key to mapped answer (shorthand for `Reply(answers=mapping)`), or a `ProviderError`; anything else raises `TypeError`.
- **The call log**, `calls`, holds one `ScriptedCall(wire_request, deadline)` per call in the order received.
  A call is logged before it is answered or raises.
- **A `ProviderError` item** is raised as it is.
  The leaf never retries: one call is one wire request.
- **A reply's answers** go through `validate_answers` with the capabilities record's `rounding`, as a wire leaf's mapped answers do.
  A scripted `Fault` or `Refusal` stands as scripted; its siblings are validated; a missing key is a fault and an extra key is dropped with an `extra-answer-key` warning.
- **The provenance** is filled as a wire leaf fills it: `elicitation="passthrough"`; `calibrated` from the capabilities record; `model_reported` from the reply or else the leaf's model; `snapshot` equal to the reported model when the capabilities record's `snapshot_pinnable` and the reply's `pinned` are both true, otherwise `None` with a `WARNING_NO_SNAPSHOT` warning; `prompt_sent` the wire request as sorted, compact JSON; `usage` from the reply or else the estimate for input and the same character rate over the raw body rendered as sorted, compact JSON for output, with both figures in `raw_usage`, and `passes` `None`; `request_id` `"<name>-<n>"`, numbered when the call is logged, so overlapping calls get distinct ids; `raw_body` the scripted answers as a JSON value, `{"answers": ...}`, the same for the same script on every run; `warnings` the validation and snapshot warnings, then the reply's own, then one `unknown-option` warning when the request carries options, since the scripted leaf knows none.
  `versions`, `client_versions` and `claimed` are the reply's; `unreported` is the reply's when given, even empty, so a test can script a profile that breaks the rule, and otherwise derived so the profile passes: the snapshot `not-pinned` when there is none, each usage figure it has none of `not-reported`, and an empty `versions` `not-applicable`.
- **A script that runs out** raises `ScriptExhausted`, an `AssertionError` and never a `ProviderError`, so a test fails rather than the loop retrying.
- **The rendering** of `prompt_sent` and `raw_body` writes a dataclass as an object tagged with its type name, a date, time, datetime or timedelta as an object tagged with its type's name holding its ISO form or its seconds, a non-finite float as an object tagged `float` holding its `repr`, so a string such as `"nan"` stays a string, bytes as base64 and a set as a sorted array.
  The recorded leaf's logs and the HTTP harness's cassettes are written in it, so a change to it bumps the format of each that holds what changed.

## The recorded leaf

`Recorded(leaf, log, mode, *, capabilities=None)`, in `idia.providers.recorded`, satisfies `Leaf` by its shape and is not a `Provider` until wrapped.
It records the calls of the leaf it wraps to the file `log` and replays them, never calling the wrapped leaf for a call it replays.

- **The key** of a call is the SHA-256, in hex, of the canonical wire request: the leaf's `name` and `model`, the state, the questions in order and the request's options, rendered as sorted, compact JSON; empty options are left out of the key and out of every rendering.
  The recorded leaf adds no warning of its own: a recording keeps the wrapped leaf's `unknown-option` warning and the replay returns it.
  Nothing else of a leaf's construction is in it, so a test keeps one log per leaf configuration and names the log for it.
  A JSON value's key order is not in the key: two wire requests that differ only in it, such as two structured instructions, share a recording.
  The key is computed over the canonical wire request, not the text the leaf sends for it, so a leaf whose serialisation changes keeps its keys and replays what its old text drew: delete its logs and record them again, since no mode replaces a recording its key already holds.
  A key's calls are kept in order; the n-th call with a key replays the n-th recording.
- **The modes.** `none` replays only: a call with no recording, or past the number of calls recorded under its key, raises `ReplayMiss`, an `AssertionError` and never a `ProviderError`, naming the key and the call number; a `log` that does not exist fails at construction.
  `once` passes every call through and records it when `log` does not exist, and otherwise behaves as `none`.
  `new` replays a call that has a recording, records one that has none, and appends a call past its key's recorded count.
- **The surface.** `name`, `model`, `capabilities` and `estimator` are the wrapped leaf's.
  With `leaf=None` the recorded leaf replays only: `capabilities` is required (`TypeError` otherwise), only mode `none` is allowed (`ValueError` otherwise), the name and model are read from the log, and the estimator is the scripted leaf's character count; a plan the real leaf's estimator shaped may differ, so wrap the real leaf in mode `none` where the plan matters.
  `capabilities=` beside a leaf raises `TypeError`, and a log recorded under another name, model or set of enabled features raises `ValueError` at construction.
  `calls` logs one `ScriptedCall` per call, before it is answered.
- **Errors.** A `ProviderError` from the wrapped leaf is stored and raised again, unchanged.
  A replayed one is a plain `ProviderError` carrying the stored `message`, `is_retryable`, `retry_after`, `status_code` and `request_id`.
- **The log** is one JSON file, `{"format": 2, "name", "model", "enabled_features", "calls": {key: [entry, ...]}}`, its header naming the configuration it was recorded under: a wrapped leaf whose name, model or enabled features differ from the header's is refused at construction with `ValueError`, and so is a `capabilities=` record, with no leaf, whose enabled features differ; each entry `{"response": ...}`, the wire response in the rendering above, or `{"error": {...}}` with the five fields.
  The raw body is kept as given, `{"json": ...}` or `{"base64": ...}`, and the raw usage as plain JSON; a log in another format, or a document that is not a recorded leaf's log, an entry included, is refused with `ValueError` when it is read.
  The whole file is written again after each recorded call, in UTF-8 with LF line endings on every platform, through a temporary file replaced in one step; a log that exists keeps its mode, a link is written where it points, and a new log takes the mode the umask gives a new file.
  A wire response's `versions`, `client_versions`, `unreported` and `claimed` are left out of the log when empty, so a log written before they existed replays with them empty, and its format stays 2; an older idia refuses a newer log that fills them.
- **Upgrading a log written before the provenance profile.** `upgrade_log(log, *, reasons=None) -> int`, in `idia.providers.recorded`, brings such a log up to the [profile](vocabulary.md#provenance) offline, with no call and no credential, and returns how many responses it changed; a second run changes none.
  Each recorded response gets a reason for every detail that holds no value, by the rules its leaf, the log's `name`, gives today: for a wire leaf (`system-one`, `openai-decisions`, `openai-chat`, `cloudflare-clef`, `cloudflare-clairic`) and `scripted`, `versions` `not-applicable`, the snapshot `not-pinned` beside its `no-snapshot` warning, and every other missing detail `not-reported`, but the passes `not-exposed` where the raw usage holds a pass count the recording idia did not read; for `gliner`, whose raw body held a `versions` map until the profile moved them out of it, `versions` and `client_versions` from that map (Python, gliner2, torch and transformers; huggingface_hub), string values only, the output tokens `not-applicable`, the other missing details as a wire leaf's, and `model_reported` and the snapshot claimed wherever they hold a value: a log cannot show whether a snapshot was a resolved commit or a revision the caller gave, so the upgrade claims one wherever a log holds it.
  Any other leaf takes the reason for each missing detail from `reasons`, which a leaf with rules of its own refuses; a detail left with a reason is never claimed.
  Nothing recorded changes: an answer, a value, a warning, the raw body and any reason, claim or version a response already holds stay as they are, error entries are untouched, and the log keeps its format and keys, rewritten only when a response changed, in one step.
  A missing snapshot's reason is always `not-pinned`.
  It raises `ValueError` and writes nothing when a response's leaf has no rules and no `reasons` are given, `reasons` are given for a leaf with rules, give `not-sent` or give a missing snapshot another reason, a response records an empty request id, a `gliner` response records no runtime version, a missing snapshot has neither a `no-snapshot` warning nor a reason, `elicitation` or `calibrated` holds no value or has a reason, or a detail would be left with neither a value nor a reason.
  A wire response holds no header, so nothing of a request's or response's headers reaches a log.
- **A replay runs through the loop** like any call: the attempts, the outcomes and `RequestFailed` are the dispatcher's, made again, with the replay's own latency.
  Give the dispatcher `RetryPolicy.without_delay()` so a recorded retry does not wait again.
  Nothing on a replayed response tells it from a live one.

## The HTTP harness, in idia's tests

idia's own `tests/_http.py` holds `HttpLeafHarness`, the leaf half's harness for a leaf on an `httpx2` client; it is test code, not part of the package, and the package imports no HTTP client.
A leaf's tests subclass it and fill its hooks: the declared `capabilities`, `capabilities_for(features)` (stated, not read back from the built leaf), `build(client, features)` (with `client=None` the leaf makes its own), `cassette_dir`, `cassette_for(kind)`, `failure_cassette`, `perturb`, `unreadable`, `refusal_body` and `candidate_body`.
`perturb`, `refusal_body` and `candidate_body` return `None` where the wire cannot make that body, and `make` then raises `CheckNotApplicable`; a leaf on an SDK overrides `sdk_retries` and `client_closed` to read its SDK client.
The base plays each exchange through an `httpx2.MockTransport` from the leaf's cassettes, and asserts that a replayed success was asked for with the cassette's method, URL and JSON body; a `Success()` with no kind is answered by whichever success cassette in `cassette_dir` that request matches, so a request packed one question per wire request is answered by a cassette recorded for it.
`handler(exchanges)` returns the mock wire's handler alone, so a router's harness can put its members' handlers behind one shared transport.
`replay_headers(headers)` returns a stored response's headers as a replay sends them, without `content-length`, `content-encoding` and `transfer-encoding`, since the body is written again, decoded; a leaf's own test wire uses it.
`sdk_retries` reads httpx2's private transport attributes, so an httpx2 release may break it.

A cassette is one JSON file per case at `tests/cassettes/<leaf>/<model>/<features>/<name>.json`, `<features>` being `base` with no opt-in feature on: `{"format": 1, "leaf", "model", "features", "request": {"method", "url", "headers", "body"}, "wire_request", "response": {"status", "headers", "body"}, "expected"}`, the wire request and the answers in the rendering above.
A cassette may carry `recorded_with`, a map from a package to the version that shaped it; no reader needs it.
`write_cassette` stores the request and response bodies verbatim, keys in the order sent and received, and the rest of the document with sorted keys.
A cassette's `wire_request` therefore holds a JSON value with sorted keys; a replay that must send a caller's key order, such as a structured instruction's, builds its request in code.
A request matches a cassette when its method and URL equal the stored ones and its body, parsed as JSON, equals the stored body with every object's keys and every array's items in the same order, since a wire's answers can move with the order of its questions; two bodies equal in value but in another order do not match, and `CassetteTransport` holds them apart.
A cassette written before `write_cassette` kept the order sent, as a probe recorded once and never written again, holds its request body with sorted keys, matches no request a leaf sends, and is replayed only by name.
A subdirectory of the features directory holds any cassette the leaf's own `*.json` glob must not read: evidence whose request Fit refuses, as Clairic's answer to a state past its context is kept in `past-context/`; and a probe whose cassettes share a body, which `CassetteTransport` refuses to hold together, as the question-order probe keeps one request sent twice and once with its questions reversed in `order/`, each of its cassettes replayed alone.
Each such directory is named by the test code that writes it, and only a test that names it replays it.
Except a hand-written cassette (below), a leaf's cassette is written only by `write_cassette`, which refuses unless `IDIA_LIVE=1`: from one call through the leaf by `record_cassette` and its recording transport, which writes nothing when the call made other than one exchange, raising `NoExchange`, an `AssertionError` with `count`, when it made none, as on a timeout or a failed connection the leaf mapped to `ProviderError`, or from a probe sent outside the leaf.
`record_cassette` takes an optional `check(side, result)`, run before anything reaches the cassette's path: the cassette is first written to `side`, a file of the same name in a fresh temporary directory outside the cassette tree, and only when `check` returns is it written at its path, through the same temporary file beside it replaced in one step, and `side` removed; when `check` raises, the cassette at its path is left as it was, or absent, `side` is kept for inspection unless the check removed or moved it, and the error is raised again.
With no `check` the cassette is written at its path directly.
`write_cassette` writes through a temporary file beside the cassette, replaced in one step, so a failed write leaves an earlier cassette as it was, or none.
It replaces the values of `authorization`, `x-api-key`, `api-key`, `cookie`, `set-cookie`, `openai-organization` and `openai-project` with `<filtered>` in both sets of headers, leaves out `user-agent` and the headers that fingerprint the account or the recording machine, and keeps the rest, the request id and the status and retry headers among them.
The filter covers headers only, not a URL or its query string; a secret elsewhere, such as Cloudflare's account id in the URL, is passed as `redact`, a map from the secret to its placeholder, replaced wherever it appears in the cassette.
The recording client has no timeout of its own, so the leaf's deadline is the only one, as on a client the leaf creates.
A run of many wire requests through one dispatcher is recorded and replayed with three more helpers.
`LeafSpy(inner)` is a leaf delegating to `inner`, its `name`, `model`, `capabilities`, `estimator`, `client` and `aclose` the inner leaf's, keeping each call's wire request with the answers returned, or `None` for a call that raised `ProviderError`, in `calls`, and each such error in `errors`; any other exception passes through and is not logged; it is wrapped before the dispatcher is built, since the dispatcher reads the leaf's estimator at construction.
`write_run_cassettes(exchanges, calls, *, leaf, path_for, redact, secrets, shapes)` pairs a recording transport's exchanges with the spy's calls one to one, refusing counts that differ, writes each at `path_for(index, wire_request)` with `redact` and reads it back, and fails when a file holds one of `secrets` verbatim or a match of one of `shapes` (by default `OPENAI_KEY_SHAPE` and `BEARER_SHAPE`; a Cloudflare run adds `ACCOUNT_ID_SHAPE`), removing every file the call wrote when any write or check fails.
`CassetteTransport(directory)` is a mock transport over every cassette directly in `directory`, refusing two with one method, URL and body at construction; it answers a request with the cassette whose method, URL and body it matches, a failure's cassette replaying its failure, keeps each path it served, in order, in `served`, keeps a request no cassette matches in `misses` before failing it, and names the cassettes never served in `unused()`.
Secrets are kept out of a committed cassette in four layers: the header filter and `redact` at write time; the exact-value check of `write_run_cassettes` at write time, the only one that catches a secret with no shape of its own, such as a Cloudflare API token outside a header; and, in the normal suite, the key and bearer shapes checked over every committed cassette and the account-id shape over every Cloudflare one.
Where a live run cannot draw a case, a stand-in may be written by hand from the wire's documented shape, its request id beginning `req_synthetic`, and the leaf's case module names it.
The cassettes of idia's own test-only stub leaf, `tests/_stub_http_leaf.py`, which no wire answers, are written by hand in the order it sends.
