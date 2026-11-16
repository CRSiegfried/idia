# Vocabulary

The shapes every part of idia speaks, the closed sets, the answer union, the rules an answer passes, and the two protocols.
Every shape is a frozen, slotted, keyword-only dataclass; a map-valued field is a read-only view.
In the shapes a caller builds, `Deadline`, `TextPart`, `ImagePart`, `State`, `Label`, `Question`, `Request` and `WireRequest`, a tuple-valued field takes a tuple: a list or other sequence is refused with `TypeError`, not converted, while a map-valued field takes any mapping and holds a read-only view of it.
Each of them refuses a field of the wrong type with `TypeError` and an empty key, instruction, label name, label description or media type with `ValueError`, each naming the field; a text part may be empty and a JSON part's value is not checked, except that a value that contains itself, or nests too deep (deeper than the interpreter's recursion limit allows from the stack where the value is built), is refused with `ValueError`, as it is in a structured instruction or an option.
The public names are those in `idia.__all__` and in the `__all__` of each module [the contributing notes](../../CONTRIBUTING.md#ground-rules-for-changes) list.

## State and question

- A **state** is the input one request is about: a tuple of parts, each a `TextPart(text)`, a `JsonPart(value)` or an `ImagePart(data, media_type=None)` holding the image's bytes.
  A JSON part's value is held read-only all the way down, each object a read-only mapping and each list or tuple a tuple, so a later change to the caller's own object reaches nothing; its contents are not checked beyond the two refusals above, and it is left out of the part's `hash()`.
- A **JSON value**, `JsonValue`, is `bool | int | float | str | tuple[JsonValue, ...] | Mapping[str, JsonValue] | None`, arrays as tuples so a value can be held read-only.
  It types a structured part, a structured instruction, raw usage, the prompt sent and the raw body.
  A read-only mapping cannot be pickled or deep-copied, so neither can a shape holding a JSON object: a `State` whose JSON part holds one, or a `Question` whose structured instruction does, raises `TypeError` from `pickle` and `copy.deepcopy`, as a `Request` or `WireRequest`, whose options are a read-only view, always has.
- **Key order.** A JSON object's keys are held and sent in the order the caller built them, at every depth, in a JSON part, a structured instruction and the options: key order is part of what the model sees, and idia never sorts it.
  `prompt_sent` holds the body in the order sent.
  `==` on the shapes compares mappings as Python does, without regard to order, so two equal shapes may be two different prompts; on the shapes a caller builds, `hash()` agrees with `==`: a JSON part's value, a structured instruction and the options are each left out of their shape's hash, so a state, a question and a request hash whatever they hold.
- A **question** is `Question(key, kind, instruction, labels=(), value_type=None, is_list=False)`.
  `labels` holds `Label(name, description=None)` entries in the order asked; for a `score` a level's position from 0 is its value.
  `instruction` is a non-empty string, or a non-empty JSON object (a mapping with string keys) or array (a tuple) for a wire API that takes structure; it is held read-only all the way down, each object a read-only mapping and each list inside it a tuple, so `json.loads` output may be passed, its scalar values unchecked, and it is left out of the question's `hash()`; a list at the top is refused, an array being a tuple.
  A `choice` needs two or more labels and a `score` one or more; an `extraction` and a `box` take none.
  A `noul` takes none, or exactly `Label(name="yes", ...)` then `Label(name="no", ...)` with at least one described: the wording of what counts as each; its answer is still one probability of yes.
  A leaf sends a question's wording or refuses it, and never drops it: its record's `carries_noul_descriptions` and `carries_structured_instructions` say which it sends, Fit refuses the rest before any call, and the leaf itself refuses a call made past Fit with a non-retryable `ProviderError`; [Providers](providers.md) says which leaf does which.
  An `extraction` declares its `value_type` and may ask for a list; no other kind does.
  A malformed question raises `ValueError`, and an instruction of the wrong type, such as a list or a number, `TypeError`.
- A **request** is `Request(state, questions, deadline, options={})`; it asks at least one question, and keys are unique within it.
  `options` maps a name to a JSON value, settings a leaf may read per request (Clairic's `reference_date`); each value is held read-only all the way down in the order given, as a JSON part's is, and they are left out of the object's `hash()`, and part of the canonical wire request when non-empty.
  A **wire request** is `WireRequest(state, questions, options={})`, one entry of a fit plan, carrying its request's options unchanged.
  A leaf treats each option key in one of three ways.
  A key it reads it records in the prompt sent.
  A key it knows but does not read, such as an option meant for another member of the same router, it takes with no effect and no warning, and records nothing for it.
  The keys it does not know it ignores, with one `unknown-option` warning naming every such key on its wire response, built by `unknown_options(options, known)`; nothing else changes, and an unknown option never reaches the wire.
- A **deadline** is an instant on the monotonic clock; build it with `Deadline.after(seconds)`, the seconds from now, and `remaining()` gives the seconds left, negative once passed, while `passed()` says whether it has passed, counting it as passed when it is within one tick of the monotonic clock, the most by which the event loop fires a timer early, so a call cut at the deadline never reads as having time left.
  `Deadline(at=...)` takes a `time.monotonic()` instant and refuses one that is not finite or is more than 30 days (2,592,000 seconds) ahead, which is what a wall-clock time such as `time.time() + 30` would be; `Deadline.after` takes a finite number of seconds of at most 30 days, a past instant allowed, so zero or negative seconds give a deadline already passed, and refuses any other value with its own message.

## The closed sets

- **Kinds**: `noul`, `choice`, `score`, `extraction`, `box`.
  The set is closed for a release and grows only by a minor release.
- **Candidate kinds**: `estimate`, `forecast`, `multi_label`; the candidate value types are `estimate` and `forecast`.
  Building a question with any of them raises `KindNotInRelease(kind, candidate=True)` before any request exists; an unknown kind raises it with `candidate=False`, and an unknown value type raises `ValueError`.
- **Value types**: `text`, `number`, `date`, `time`, `datetime`, `duration`, each held by the matching Python type (`str`; `int` or `float`, not `bool`; `date`; `time`; `datetime`; `timedelta`).
  A list value is a tuple of them.
- **Elicitations**: `passthrough`, which every decision model reports, and `logprobs`, the chat leaf's.
  The candidates `structured_output`, `verbalised` and `sampled_n` cannot build a chat leaf: construction raises `ElicitationNotInRelease(elicitation, candidate)`.
  `check_elicitation(name)` returns an admitted name or raises it.

## Capabilities

A `Capabilities` instance is a capabilities record, a leaf's or a composite's derived one; in this spec "record" alone means it.
`Capabilities` states every field with no default on any limit.
A limit is a non-negative `int`, `UNBOUNDED` (the wire API documents no cap) or `None` (unknown); the constructor refuses any other type, a `float` included, with `TypeError` naming the field, and a `bool` or a negative `int` with `ValueError`.
`min_levels` and `min_image_side` are each a non-negative `int` or `None`, refused the same way.
Four fields bear only on a state with an image: `max_image_bytes`, each image's raw bytes, and `max_total_image_bytes`, the sum over the state's images, are limits; `min_image_side`, the fewest pixels on each side of an image, is a non-negative `int` or `None`, refused as `min_levels` is, 0 meaning no minimum; and `image_media_types` is a set of media types or `None` (unknown), held frozen.
The constructor refuses a media type that is not lower-case and canonical, `image/jpg` among them, naming the form to write; on a leaf's record it refuses an empty `image_media_types`, or a `max_images`, `max_images_per_question`, `max_image_bytes` or `max_total_image_bytes` of 0, beside the image modality, any of which would refuse every image.
`Rounding(probability_decimals, score_decimals)` gives the decimals the provider rounds to, `None` for unrounded.
`carries_noul_descriptions` says whether the leaf sends a `noul`'s `yes` and `no` descriptions, and `carries_structured_instructions` whether it sends an instruction that is a JSON object or array; each is a plain `bool`, and false means Fit refuses a request that uses it before any call.
The constructor refuses `truncates_silently=True` with `max_input_tokens=None` on a leaf's record; a record built with `composite=True` is a derived one and skips that refusal.
`feature_kinds: Mapping[str, frozenset[Kind]]` and `feature_modalities: Mapping[str, frozenset[Modality]]` are required, read-only maps from an opt-in feature to the kinds, or the modalities, it adds on this model, whether or not it is on, so they are the same for a model whatever is enabled.
A feature absent from a map adds none; a record with no opt-in feature has both empty.
Each map must be a mapping and each of its values a set or frozenset, or the constructor raises `TypeError` naming the entry, as `Capabilities.feature_kinds['images'] must be a set, got str`.
The constructor refuses a map that has a key outside `opt_in_features`, that maps a feature to the empty set, or that maps a feature in `enabled_features` to a kind not in `kinds` or a modality not in `modalities`.
An off feature may map to a value the record already has, since another feature or the model may supply it.

## Answers

The answer union is `LabelAnswer | NoulAnswer | ExtractionAnswer | BoxAnswer | Refusal | Fault`.
Nothing in it was computed by idia.

- `NoulAnswer(probability, confidence=None)`: the provider's probability of yes, as given; no second value.
  `noul_complement(probability)` gives the probability of no: the exact decimal difference `1 - probability`, of the probability as its repr writes it, converted to the nearest float and never rounded, so `0.7` gives `0.3` where the float subtraction gives `0.30000000000000004`.
  It raises `TypeError` for a bool or a value that is not a number and `ValueError` for one that is not finite or lies outside 0 to 1; nothing stores it on an answer, and a leaf never calls it.
- `LabelAnswer(probabilities, confidence=None, chosen=None, expected_level=None, echo=None, echo_of=None)`: `probabilities` pairs each label name with its probability in wire order; `echo` is the labels or legend the wire echoed back, and `echo_of` says which.
- `ExtractionAnswer(value, guess=None, confidence=None, p_null=None, span=None)`: `value` of the declared type or `None`; `guess` the provider's own, never filled by idia and never moved into `value`; `p_null` the provider's probability that no value exists, `None` where it gives none (a list, for one), never computed by idia; `span` one per item for a list.
- `Span(part, start, end)`: `part` the index, in `state.parts`, of the text part the value was found in; `start` and `end` 0-based, end-exclusive code-point offsets, so `text[start:end]` is the span's text.
- `BoxAnswer(region, confidence=None)` with `Box(top, left, bottom, right)` as fractions of the image.
- `Refusal()`: the provider declined; an answer, never an exception.
- `Fault(key, failures, mapped)`: every `RuleFailure(rule, observed, bound)` the answer broke, and the answer as mapped, or `None` when nothing could be mapped.
  A fault is scoped to one question; its siblings stand.
- `off_label_mass` and `derived` on a label or noul answer are the chat leaf's; a decision model's answer carries `None` and an empty set.

## Validation

A leaf calls `validate_answers(questions, mapped, rounding)` after mapping and before the answers enter its wire response.
A question whose key is absent gets a `Fault` with `Rule.missing_key`; a key with no question is dropped with a `WARNING_EXTRA_ANSWER_KEY` warning; every other entry goes through `validate(question, mapped, rounding)` once.
A label answer that stands with its chosen label below the maximum adds a `WARNING_CHOSEN_BELOW_MAX` (`chosen-below-max`) warning whose field is the question's key, before any `extra-answer-key` warning.
A `Fault` or `Refusal` the leaf built passes through as it is; a fault keyed to another question is a leaf bug and raises `ValueError`.

`validate` runs every rule for the kind.
An answer that passes comes back as the same object; one that fails becomes a `Fault` naming every rule it broke.
Nothing is renormalized, rounded or corrected.

- **Shape**: the answer has the kind's shape (`kind_shape`).
- **`choice` and `score`**: the label names are those asked, in order, none added or repeated (`labels_mismatch`); an echo equals the names asked, or the descriptions asked with a label's name standing in where it has no description (`echo_mismatch`); every probability is finite (`not_finite`) and in [0, 1] (`probability_range`); the sum lies within `1e-6 + n · 0.5 · 10^-d` of 1 for `n` labels asked and `d` probability decimals, the second term zero when unrounded (`sum_tolerance`); a reported `chosen` has the maximum probability, ties allowed, or lies below it by at most one step of the declared probability decimals, `10^-d + 1e-6`, or `1e-6` when unrounded (`chosen_not_max`), since a provider may round after it chooses; the chosen label is kept as given.
- **`score` expected level**, where reported: within `1e-6 + 0.5 · 10^-s + 0.5 · 10^-d · n(n-1)/2` of the sum of each level index times its probability, for `s` score decimals, each term zero when its decimals are `None` (`expected_level`).
- **`noul`**: the probability is finite and in [0, 1] (`probability_range`).
- **`extraction`**: `value` and `guess` have the declared type or are `None`, every item for a list (`value_type`); `p_null`, where given, is in [0, 1] (`p_null_range`).
  The span is not checked.
- **`box`**: each edge is in [0, 1] (`box_range`); top is at most bottom and left at most right (`box_order`).
- **Any `confidence`** present is in [0, 1] (`confidence_range`).
- **Every probability, confidence, `p_null` and box edge** is a number, not a bool, and finite (`not_finite`, observing a non-number as its type's name, such as `bool`, and an int too large for a float as `int beyond float`); the sum, the chosen label, the expected level and the box order are not judged over a value that is not a number.
- **A score's `expected_level`** present is a number, not a bool, that a float can hold (`not_finite`, observed as above), and is then judged against the probabilities.

For a range rule `observed` is the value and `bound` the limit crossed; for a tolerance rule `observed` is the deviation and `bound` the tolerance; for an order rule the two values out of order.
`Rule.candidate_shape`, `empty_logprobs`, `label_absent_from_logprobs` and `off_label_mass_above_bound` are set by a leaf itself; `validate` never checks the off-label bound.

## Responses

- `WireResponse(answers, elicitation, calibrated, model_reported=None, snapshot=None, prompt_sent=None, usage=None, request_id=None, raw_body=None, warnings=(), versions={}, client_versions={}, unreported={}, claimed=frozenset(), extras={})`: what a leaf returns from one call.
  The raw body travels no further.
- `Outcome`: one per wire request, in plan order: the `keys` it carried, in the order the wire request sent them, its `error` (a `ProviderError` or `None`), the leaf's `provider` and `model_requested`, `model_reported`, `snapshot`, `elicitation` and `calibrated` (unknown on a failed outcome), `prompt_sent`, `usage`, `latency`, `started_at`, `request_id`, `failed_attempts`, `warnings`, `rounding`, its leaf's record's, so a composite's outcome gives the precision of the member that answered, the provenance fields below and its [extras](#extras).
  `attempts` is the failed attempts plus one when answered, 0 for a wire request never sent.
  `started_at` is wall-clock seconds since the epoch; `latency` and a failed attempt's `duration` are monotonic durations.
  `prompt_sent` is the supported way to see what was sent: the body or messages as the leaf sent them, `None` when the request was never sent or the wire API hides it.
  A leaf whose wire API restricts question ids may send ids other than the caller's keys, for every question of a wire request or for none; the rules are each leaf's, in [Providers](providers.md), and the id at position *i* of a wire request always stands for `keys[i]` of its outcome.
  For Clef asked `q:match/intent` and `q:match/tone`, which its id pattern refuses, `outcome.prompt_sent["questions"]` holds `q0` and `q1`, and `q1` is `outcome.keys[1]`, `"q:match/tone"`; `Response.answers` and a `Fault`'s key are always the caller's.
- `Response(answers, outcomes, warnings=())`.
  It has no provenance of its own; the outcomes hold it.
- `Notice(code, message, field=None)`: something the caller should know that is not a fault, held in `warnings`.
  The codes the core raises are published as constants: `WARNING_EXTRA_ANSWER_KEY` (`extra-answer-key`), `WARNING_UNCHECKED_LIMIT` (`unchecked-limit`), `WARNING_OVERFLOW_POLICY` (`overflow-policy`), `WARNING_PROVIDER_TRUNCATED` (`provider-truncated`, a leaf's report that the wire truncated the state), `WARNING_UNKNOWN_OPTION` (`unknown-option`, field `options`), `WARNING_NO_SNAPSHOT` (`no-snapshot`) and `WARNING_LABEL_ABSENT_FROM_LOGPROBS` (`label-absent-from-logprobs`), raised by a chat leaf, and `WARNING_CHOSEN_BELOW_MAX` (`chosen-below-max`, field the question's key).
- `FailedAttempt(provider, model_requested, error, request_id, started_at, duration, usage=None)`.

### Provenance

Every outcome states one profile of detail, each a value or a reason why it is not.
The details are `DETAILS`, the members of `ProvenanceDetail`: `model_reported`, `snapshot`, `versions`, `elicitation`, `calibrated`, `prompt_sent`, `usage.input_tokens`, `usage.output_tokens` and `request_id`; `None`, an empty `versions` and a `None` `usage` hold no value.
`provider` and `model_requested` are always filled, latency, start time and warnings are the dispatcher's, and precision is the outcome's `rounding`, so none of them is a detail.
These names are shared with ommat, which calls `snapshot` its revision, `versions` its runtime and `prompt_sent` its prompt.

`WireResponse` and `Outcome` each carry four provenance fields, read-only and empty by default:

- `versions: Mapping[str, str]`: the versions of the runtime that ran the model, package to version, with `python` for the interpreter; empty for a wire API, whose runtime is the service's own; a per-call backend fingerprint a wire API returns, such as OpenAI's `system_fingerprint`, is no version and stays in the raw body.
- `client_versions: Mapping[str, str]`: the versions of the client libraries that made the call; a leaf fills it and never sets the key `idia`, which the dispatcher adds, its own version, on every outcome, its entry winning.
- `unreported: Mapping[ProvenanceDetail, UnreportedReason]`: the reason for each detail that holds no value, one of `REASONS`: `not-reported`, the provider or runtime gives none; `not-pinned`, no fixed version was asked for or resolved, the snapshot's reason exactly when the `no-snapshot` warning is given; `not-sent`, the wire request failed, sent or not, which only the dispatcher gives; `hidden`, the provider's service withholds it; `not-applicable`, the detail has no meaning for this source or its kind of model, such as `versions` for a wire API or the output tokens of an encoder; `not-exposed`, a library between the provider and the caller has it but does not make it public.
- `claimed: frozenset[ProvenanceDetail]`: the details whose value the source asserts but cannot verify, such as a model id a leaf states as its own constant, or a revision a caller names for a model it loaded itself.

Either constructor raises `ValueError` for a key of `unreported` or `claimed` outside `DETAILS`, a reason outside `REASONS`, a detail both unreported and claimed, a detail that holds a value and has a reason, or a claimed detail that holds none, and `TypeError` for a field of the wrong shape.
That every detail with no value has a reason is checked by the conformance suite, not the constructor, so a hand-built response still builds.
An answered outcome carries its wire response's `versions`, `unreported` and `claimed` as given, and its `client_versions` with `idia` added; a failed outcome, sent or not, has `not-sent` for each detail that holds no value, a request id its error carries staying a value, and claims nothing.

### Extras

`WireResponse` and `Outcome` each carry `extras: Mapping[str, JsonValue]`, empty by default: the figures the wire reports beyond the shared vocabulary, such as Clairic's passes as `{"passes": 2}`.
Its keys are the leaf's own, and the [providers spec](providers.md) lists each leaf's; an empty `extras` means the wire reported nothing more.
It is no provenance detail, so it has no reason in `unreported` and is never claimed.
It is held read-only at every depth in the order built, each object a read-only mapping and each array a tuple, and either constructor raises `TypeError` for an `extras` that is not a mapping and `ValueError` for a key that is not a string, a value outside the JSON domain or a float that is not finite, anywhere inside it; no size is bounded.
An answered outcome carries its wire response's `extras` unchanged; a failed outcome, sent or not, has none, and a `FailedAttempt` carries none.
A figure is read from `extras` by the key its leaf documents, never from `raw_usage`; a figure every source turns out to report is promoted into the shared vocabulary by a deliberate change, not read from `extras` by convention.

### Usage

`Usage(input_tokens, output_tokens, raw_usage={})`.
`input_tokens` and `output_tokens` are as the wire reports them, each `None` when it reports none; `raw_usage` is the wire's usage object as given, its own keys unchanged.
A decision model that returns probabilities or values and no text reports `output_tokens` 0, and 0 is kept as 0.
A 0 does not mean an answer is missing: a question with no answer gets a `Fault`, a wire request that failed has an outcome with its `error` set, and `RequestFailed` is raised only when no wire request answered.

## Errors

`IdiaError` is the base.
`ProviderError(message, *, is_retryable=False, retry_after=None, status_code=None, request_id=None)` is the one error a leaf raises; `DeadlineExceeded` is a non-retryable one that Dispatch, never a leaf, puts on the outcome of a wire request the deadline ended: one never sent because the deadline passed, or one whose next retry the deadline could not fit, its failed attempts listed with their own errors; the dispatcher builds it when the deadline has passed or a retry's delay would end past it, and a limiter raises it when its wait would end past the deadline.
A `ProviderError`'s message, and so its `str()`, has every key-shaped substring, an `sk-` API key (`sk-` and twenty or more key characters) or a bearer token written out (the word `Bearer`, in any case, then white space and sixteen or more token characters), replaced by `<filtered>` when the error is built; its other fields, the request id among them, are kept as given, and a `RequestFailed`, which quotes the first failed outcome's error, therefore carries no such key either.
Each wire leaf also filters the credentials it was built with out of every error text it builds from the wire or its client, before any cut: Cloudflare's API token, System One's API key, and the OpenAI client's `api_key` and `admin_api_key`, read when the error is built; a credential shorter than eight characters is left as it is.
Any other credential of another shape passes as written, and only the text is filtered: an exception the error was raised from, held in its `__context__`, is not.
`RequestFailed(response)` carries the full response when no wire request answered.
Its message names the first failed outcome in `response.outcomes`: `no wire request answered: <provider> <model_requested>: <error> (1 outcome failed)`, or with `(<n> outcomes failed; the first shown)` when several failed, `<error>` being the error's text, or its type name when the text is empty.
A hand-built response with no failed outcome gives `no wire request answered (1 outcome)`, or `no wire request answered (<n> outcomes)` for any other count.
`DoesNotFit(message, *, check, observed, limit, state_overflow=False, off_features=frozenset())` is raised before any call when a request does not fit; [Fit](fit.md) gives its fields.
`KindNotInRelease` and `ElicitationNotInRelease` are above.

## The two protocols

Both are `runtime_checkable` and nothing is subclassed to satisfy them.
Their data members are read-only: a plain attribute, a property or a frozen dataclass field satisfies each.

- `Leaf`: `name`, `model`, `capabilities`, `estimator` and `async call(wire_request, deadline) -> WireResponse`.
  It sends one wire request in one call, with the deadline as its timeout, never retries, and raises `ProviderError` and nothing else.
  A leaf is not a provider until the core wraps it.
- `Provider`: `capabilities` and `async answer(request) -> Response`.
- `TokenEstimator`: `estimate(state, questions) -> TokenEstimate(tokens, margin)`, the type every leaf's `estimator` satisfies.
  `questions` may be empty, meaning the state alone; preflight asks for that to check the state for overflow.
  `margin` is a fraction of `tokens`, at least 0, and preflight checks `ceil(tokens * (1 + margin))`.
  `TokenEstimate` refuses a `tokens` that is not an `int` or a `margin` that is not a number, a `bool` included, with `TypeError` naming the field, and a margin that is not finite or either value below 0 with `ValueError`.
