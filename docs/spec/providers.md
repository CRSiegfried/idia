# Providers and capabilities

The wire leaves idia ships, the HTTP base they share, and the capabilities record each declares.
A wire leaf is an optional extra: `idia.provider(name, **options)` imports it behind a guard that names the extra to install, builds it from the options and wraps it in a `Dispatcher`.
`idia.leaf(name, **options)` builds the same leaf through the same guard and returns it unwrapped, and `idia.provider_names()` lists the names in the table below.

| Name | Leaf | Extra | Dependency |
|---|---|---|---|
| `system-one` | `SystemOne` in `idia.providers.system_one` | `system-one` | `httpx2` |
| `openai-decisions` | `OpenAIDecisions` in `idia.providers.openai_decisions` | `openai` | `openai`, `httpx2` |
| `openai-chat` | `OpenAIChat` in `idia.providers.chat` | `openai` | `openai`, `httpx2` |
| `cloudflare-clef` | `Clef` in `idia.providers.cloudflare` | `cloudflare` | `httpx2` |
| `cloudflare-clairic` | `Clairic` in `idia.providers.cloudflare` | `cloudflare` | `httpx2` |

## The HTTP base

`HttpLeaf(*, base_url, request_id_header, client=None)`, in `idia.providers.http`, is what a leaf over a raw HTTP API may inherit.
Inheriting it is a convenience: satisfying `Leaf` is structural.

- **The client.** An injected `httpx2.AsyncClient` is used as it is and never closed by the leaf.
  With none, the leaf makes one with `created_client()`, which has no transport-level retries and no timeout of its own, and `aclose()`, or leaving `async with`, closes it.
  A client is bound to the event loop it is first used on.
  A created client follows the running loop: on another loop the leaf drops it unclosed, since its loop may be gone, and creates a fresh one; `aclose()` closes the current one, or drops it when it was last used on another loop.
  An injected client is used on the loop it was first used on; a call on another loop raises a non-retryable `ProviderError`, `this client was first used on another event loop; pass a client per loop or let the leaf create one`, before anything is sent.
  After `aclose()` a leaf raises a non-retryable `ProviderError`, `the leaf is closed`, on every call; a leaf whose client it created creates no further client.
  A client that reports itself closed (`is_closed`), injected or created, is refused the same way before anything is sent: a non-retryable `ProviderError`, `the client is closed`.
  `client` is read-only.
- **One POST per call.** Each request carries `User-Agent: idia/<version>`, set per request so an injected client's own headers stay as they are.
  The call runs under `within(deadline, awaitable)`: a deadline already passed, as `Deadline.passed()` reads it, raises a retryable `ProviderError` before anything is sent, and one reached while waiting raises the same.
  Nothing is retried below the leaf.
- **The status table** is `status_error(status, headers, content, request_id) -> ProviderError`, behind `map_status(response)`, which a leaf may override.
  `within`, `status_error`, `retry_after`, `created_client`, `follow_the_loop`, `on_this_loop`, `malformed_request` and `client_failed`, and `read_object` and `unreadable` for a success's body, serve a leaf that does not inherit the base.

| What came back | `is_retryable` |
|---|---|
| 408, 409, 429, any 5xx (529 included) | yes |
| any other status that is not a success (400, 401, 403, 404, 413, 415, 422, ...) | no |
| a connection error, a transport timeout, the deadline reached | yes |
| a request the client refuses as malformed before anything is sent: an illegal header value, a value it cannot encode, a URL with no or another scheme, or one it cannot parse; the message names the error's class, an illegal header or a value it cannot encode by its class alone | no |
| any other request error | no |
| a bare `RuntimeError` the client lets out mid-call, the backstop for a connection that died under the client, as from a pooled connection whose event loop has closed (a closed client is refused before sending; a subclass propagates) | yes |
| a success whose body is not JSON or not a JSON object | no |

- **`retry_after(headers)`** is `retry-after-ms` divided by 1000 when present and readable, else `retry-after` as seconds or as an HTTP-date (a date in the past waits nothing), else `None`.
- **The error** carries the status, the request id from the leaf's `request_id_header`, and a message naming the wire's `error`, `message` or `detail`, cut to a few hundred characters.
  It keeps no body.

## System One

`SystemOne(*, model="jev-1.14.0", api_key=None, base_url="https://api.typesafe.ai", client=None, long_context=False, images=False, capabilities=None)`, registered as `system-one`, speaks TypeSafe's System One API for one Jev model.
Its `name` is `system-one` and its `model` the one it was built for.

- **Construction refuses** an unknown model and a missing key (`api_key`, else `TYPESAFE_API_KEY`), each with `ValueError`.
  The models known are `jev-1.13.0`, `jev-latest`, `jev-1.14.0` and `jev-preview`; the default is `jev-1.14.0`.
  `jev-latest` is a moving alias, offered without `box` since the version it names may change; the response's `model` names the version that answered, and a caller who needs `box` pins `jev-1.14.0`.
  `capabilities` replaces the declared record for a caller who knows better; it must agree with the feature flags, which alone decide the headers sent.
- **The request** is `POST /v1/systemone` with `Authorization: Bearer <key>`, uncompressed JSON whose keys are `model`, `state`, `questions` and, with an image, `assets`, in that order, built the same way every time.
  `long_context` adds `TypeSafe-Danger-Zone-Long-Context: true`; `images` adds `TypeSafe-Danger-Zone-Images: true`.
- **The state**: one part is its own value (text as a string, JSON as its value); several are an array in order.
  An image becomes the text `<image-N>`, numbered from 0, and `assets` maps that text to `{type: "image", format, data_base64}`.
  The format is `png` or `jpeg`, the type Fit judged: the declared media type, lower-cased with `image/jpg` read as `image/jpeg`, when PNG or JPEG, else the type the bytes show; a leaf called directly with another image sends its declared type with `image/` removed, else the type its bytes show, else an empty format, and the wire refuses it.
  The leaf refuses before sending, with a non-retryable `ProviderError`, a state that cannot be written as JSON (a non-finite number, a value JSON has no form for) and a state with more than one image under a `box` question, since a box answers about one image: no record field says either.
- **An image below 280 pixels on a side** does not fit: the record declares `min_image_side` 280, and Fit refuses it before any call.
  An image whose size cannot be read is sent with an `unchecked-limit` warning, and the wire judges it.
  The minimum was measured on `jev-1.14.0` with PNGs: 280 by 280 and 280 by 320 either way were answered, 279 by 279 and 279 by 320 either way refused with 400 and "Invalid image dimensions.", so the rule is per side, not by area.
  JPEG was not measured; the same minimum is applied to it.
  The 16 MiB per image and 32 MiB for all images are declared and checked on the raw bytes; the documented maximum side, 1,568, is checked by no one, so the wire refuses or resizes a larger image.
- **The questions**, keyed as asked and in request order: a `noul` and a `box` send `{type, instructions}`, a described `noul` adding `criteria`, `true` for its `yes` description and `false` for its `no`, an undescribed side left out; a `choice` adds `criteria`, each label's name to its description or `null`, in label order; a `score` adds `criteria`, the list of each level's description or, lacking one, its name.
  `instructions` is the question's instruction as it is, a string or a JSON object or array; the recorded `wording` cassette on each model holds a described `noul` and an object instruction.

**Reading the response.** The body must hold a non-empty string `model`, an object `usage` and an object `answers`, or the call raises a non-retryable `ProviderError`.
Each answer is mapped to its kind's shape, then `validate_answers` runs over them all with the record's rounding.

- `noul`: `NoulAnswer(probability=noul)`, nothing else.
- `choice`: `LabelAnswer` with the probabilities named by label in the order asked, `chosen` the wire's `choice`, and its `confidence`.
- `score`: the probabilities keyed by level index become the level names in order; `expected_level` is the wire's `score`; `echo` is the `legend`'s values in index order with `echo_of="descriptions"`; and its `confidence`.
- `box`: `BoxAnswer(Box(top, left, bottom, right), confidence)`, the confidence `None` where the wire gives null.
- A label the wire left out stays out and one it added is kept after the asked ones, so validation faults the answer.
  An answer whose `type` is not its question's, or whose fields are not the kind's wire shape (a number too large for a float included), is that question's `Fault` with `Rule.kind_shape`.
  A key with no question is dropped with a warning.

**Provenance.** `elicitation="passthrough"`; `calibrated` from the record; `model_reported` the body's `model`, which is also the `snapshot` when it has the form `jev-<major>.<minor>.<patch>`, otherwise no snapshot and a `no-snapshot` warning; `usage` the integer `input_tokens` and `output_tokens` with the body's `usage` as `raw_usage`; `request_id` from `x-typesafe-request-id`; `raw_body` the parsed body; `prompt_sent` the body sent with each asset's `data_base64` replaced by `bytes`, its size.

**The record**, per model and feature:

| Field | Value |
|---|---|
| `kinds` | `noul`, `choice`, `score`; `box` as well with `images` on `jev-1.14.0` and `jev-preview` |
| `modalities`, `image_carriers` | text and JSON; with `images`, image too, carried by `placeholder` |
| `opt_in_features` | `long_context` and `images`, on every model |
| `feature_kinds`, `feature_modalities` | `images` adds the image modality on every model, and `box` on `jev-1.14.0` and `jev-preview`; there is no `feature_kinds` entry on the others, and `long_context` is in neither map. The maps are the same whatever is enabled |
| `max_input_tokens`, `max_question_tokens` | 64,000 and 32,000; with `long_context`, 262,000 and still 32,000. These are conservative readings of the documented "64K" and "262K"; the per-question limit under long context is not confirmed |
| `max_body_bytes` | 33,554,432 |
| `max_images`, `max_images_per_question` | `UNBOUNDED` with `images`; 0 without |
| `max_image_bytes`, `max_total_image_bytes` | 16,777,216 and 33,554,432, with or without `images` |
| `min_image_side`, `image_media_types` | 280; `image/png` and `image/jpeg`, with or without `images` |
| `max_labels`, `max_levels`, `min_levels` | 255, 10, 1 |
| `max_questions`, `max_total_options`, `key_patterns` | `UNBOUNDED`, `UNBOUNDED`, none |
| `confidence` | `choice`: `normalized_top_probability`; `score`: `mad_concentration`; `box`, where offered: `undocumented` (the model gives it, by no published formula) |
| `rounding` | two decimals for probabilities and for the expected level, as the recorded responses show |
| `calibrated`, `snapshot_pinnable` | true |
| `carries_noul_descriptions`, `carries_structured_instructions` | true |
| `returns_refusal`, `recounts_state_per_question`, `truncates_silently` | false |

**The estimator**, `SystemOneEstimator(*, image_tokens=320, margin=0.1)`: 260 tokens of overhead, plus the state's JSON at 6 characters a token, plus the questions' JSON at 4, plus `image_tokens` per image.
On the recorded requests the wire counted between 3% fewer and 8% more input tokens for text, and about 300 for a 320 by 320 image, the only size whose cost was measured; the probes of the minimum side, at 280 by 280 and 280 by 320, used 498 and 538 input tokens in all, within the estimate and its margin.

**What the wire does past a limit.** Eleven score levels are refused with 422 and a `detail` list naming the field; a request over the token limit with 400 and `{"detail": {"error_type": "max_tokens_exceeded"}}`.
Neither is retried.
System One has no status mapping of its own, so every 429 takes the table's default and is retryable, a quota exhausted included, since no recorded response shows how the wire marks one; the retries end at the deadline.

## OpenAI

Both OpenAI leaves speak through the `openai` SDK, its retries always zero.
An injected `AsyncOpenAI` is used through `with_options(max_retries=0)` and never closed; without one, the leaf creates one with `api_key` or else `OPENAI_API_KEY` (neither raises `ValueError`) over `created_client()`, closed by `aclose()`, which is safe to repeat.
The client follows the event loop as `HttpLeaf`'s does: a created client is made afresh on another running loop and the old one dropped unclosed, an injected one on another loop raises the same non-retryable `ProviderError` before any SDK call, an injected or created client whose `is_closed()` is true raises a non-retryable `ProviderError`, `the client is closed`, before any SDK call, and a call after `aclose()` raises `the leaf is closed`.
Each call goes through the SDK's `with_raw_response` under `within(deadline, ...)` and the raw body is read as a whole, as above.
The SDK's errors map to `ProviderError`: a status error through the status table, except that a 429 whose error `code` is `insufficient_quota` is not retryable; a connection error or timeout is retryable, but one whose cause is a request the client refuses as malformed before sending, as in the status table above, is not, and nor is a value the SDK cannot encode, a key among them; any other SDK error is not; a bare `RuntimeError` the SDK lets through during the call is retryable, the backstop for a connection that died under the client, and such a call may have reached the wire; a subclass propagates.
The request id is `x-request-id`.
The key a 401 echoes, which the wire masks itself (`sk-idia-***alid`, its first characters, stars and its last four), is not key-shaped and passes as sent; a request id quoted in an error body is kept in the message, and the error's `request_id` holds the header's.

**The decisions leaf**, `OpenAIDecisions(*, model="gpt-6-luna", api_key=None, client=None, max_questions=200, capabilities=None)`, named `openai-decisions`, sends one `POST /v1/decisions`.
Any model name is sent; the record is `gpt-6-luna`'s.

- **The body**: `model`; `input`, one user message whose `content` holds a part per state part in order: text as `input_text`, JSON as `input_text` holding compact JSON, an image as `input_image` with a base64 data URL whose media type is the one Fit judged: its declared type, lower-cased with `image/jpg` read as `image/jpeg`, when PNG or JPEG, else PNG or JPEG read from its bytes (a leaf called directly with another image sends its declared type, else `application/octet-stream`, and the wire refuses it); `questions`, each named `q<i>` in the order asked.
  A `noul` is `{type: "predicate", name, instructions}`, its instruction sent as it is; a described `noul` and a structured instruction, which its record does not carry and Fit refuses, are refused by the leaf too, for a call made past Fit, before sending with a non-retryable `ProviderError` naming the question, since a `predicate` has no field for the descriptions and `instructions` is a string; a `choice` adds `choices`, `{value: <label name>, description?}` per label, string values only; a `score` adds `levels`, `{label: <level name>, description?}` per level.
- **Reading**: the body needs an `answers` list, a non-empty string `model` and an object `usage`, or the call raises a non-retryable `ProviderError`.
  Each answer is keyed back by its echoed name; one whose name was not assigned, or that has none, is dropped with an `extra-answer-key` warning, so its question faults as missing, and a second answer repeating a name is dropped with the same warning, the first standing.
  A `refusal` is a `Refusal`.
  A `predicate` is `NoulAnswer(probability)`.
  A `choice` is a `LabelAnswer` with its probabilities in label order, `chosen` the wire's `choice` and its `confidence`; a value that is not a string is that question's `Fault` with `kind_shape`, its type named.
  A `score` orders its probabilities by level index and names each by the level asked at that index (an index out of range keeps its number, for validation to fault), `expected_level` is the wire's `score`, and the echoed `label`s are `echo` with `echo_of="names"`.
  Any other wrong shape is a `kind_shape` fault.
- **Provenance**: `passthrough`; `calibrated` false; never a snapshot, with a `no-snapshot` warning on every response, since the wire echoes the alias; `prompt_sent` the body with each image's data URL replaced by its media type and size; usage the integer `input_tokens` and `output_tokens` as the wire reports them, with the whole `usage` object raw; Decisions returns probabilities over the labels asked and no text, so it reports `output_tokens` 0 on every response, and that 0 does not mean an answer is missing (see [`Usage` in vocabulary.md](vocabulary.md#usage)).
- **The record**: kinds `noul`, `choice`, `score`; text, JSON and image, carried by `parts`; `max_questions` 200 (the option replaces it), `max_images` and `max_images_per_question` 128, `max_labels` 255; `image_media_types` PNG and JPEG, the types the leaf sends, so an image that is neither by its declared type nor by its bytes does not fit and is refused before any call; `max_image_bytes`, `max_total_image_bytes` and `min_image_side` unknown, so a request with an image goes with an `unchecked-limit` warning for each; `max_input_tokens`, `max_question_tokens`, `max_body_bytes`, `max_levels`, `max_total_options` and `min_levels` unknown, so a request goes with `unchecked-limit` warnings; `returns_refusal` true; `calibrated`, `snapshot_pinnable`, `recounts_state_per_question`, `carries_noul_descriptions` and `carries_structured_instructions` false; `truncates_silently` unknown; confidence `undocumented` for `choice` and `score`; rounding two decimals for both, as the recorded responses show.
- **The estimator**: 100 tokens of overhead, the state's text at 6 characters a token, 90 tokens per question plus its JSON at 4, 1,000 tokens per image, margin 0.1.
  On the recorded requests the wire counted the state once, and each question 60 to 110 tokens beyond its text; the image figure is unmeasured.
- **What the wire did.** It echoed every name and every level `label` as sent, and refused some questions while answering their siblings.
  It answered 201 questions with 200; the record still declares 200, the documented SDK cap, which a caller may raise.
  On `gpt-6-luna` an identical body answered identically, and four unsaturated questions sent in three orders, each alone, and reversed under their first names moved no answer at the wire's two decimals.
  An unknown model name is sent as given and answered 404 `model_not_found`, a non-retryable `ProviderError`; it is not a credential error, so the plan's other wire requests are still sent.
- **The JSON part's text.** A JSON part is sent as compact JSON, `separators=(",", ":")`, raw Unicode, its keys in the value's own order and unsorted, so a value sends the same text each time, and two equal values whose keys stand in different orders send different texts.
  On `gpt-6-luna`, one `noul` question over the state `{"t:state": {"request": "what did Kim say about the invoice?"}}` answered 0.83 through this leaf's compact text, and 0.83 again with that text sent by a direct call as a bare input string; the same state written with `json.dumps`' default separators, `", "` and `": "`, and sent the same direct way, answered 0.62, one call each; without the `t:state` wrapper the question was refused either way.
  A recording made with other serialisation does not stand for a request sent through this leaf.

**The chat leaf**, `OpenAIChat(*, model, api_key=None, client=None, elicitation="logprobs", max_off_label_mass=None, capabilities=None)`, named `openai-chat`, reads label probabilities from a chat model on Chat Completions by the `logprobs` protocol in [Elicitation](elicitation.md).
`model` is required.
The name is checked by `check_elicitation`: a candidate elicitation raises `ElicitationNotInRelease`; `passthrough` raises `ValueError`, as does a bound outside [0, 1].
A `capabilities` record is refused with `ValueError` when it contradicts the protocol: `calibrated` true, `carries_structured_instructions` true, a confidence map, rounding other than none, `max_questions` other than 1, a label or level cap that is not a whole number of at most 20, so `None` and `UNBOUNDED` are refused too, the image modality, or image fields other than 0, 0, 0 and no media types.

- **The call** is one `POST /v1/chat/completions` per wire request, which holds one question (more raise a non-retryable `ProviderError` before sending): `model`, the messages, `logprobs: true`, `top_logprobs: 20`, `temperature: 0.0`, `max_completion_tokens: 1`, no `logit_bias`.
  A state's text parts are sent in order, JSON as compact JSON; an image, more labels than 20, or a structured instruction, which its record does not carry, is refused before sending, for a call made past Fit.
  A described `noul`'s descriptions go on its letter lines, `A. Yes: <yes description>` and `B. No: <no description>`.
- **Reading**: the body needs a non-empty `choices` list, a non-empty string `model` and an object `usage`, or the call raises a non-retryable `ProviderError`.
  The first choice's `logprobs.content[0].top_logprobs` is read; a wrong shape there is a `kind_shape` fault.
- **Provenance**: `logprobs`; `calibrated` false; `model_reported` the body's `model`, the `snapshot` when it ends in a date `-YYYY-MM-DD`, otherwise none and a `no-snapshot` warning; `prompt_sent` holds the messages, the model, the temperature, `max_completion_tokens`, the top-k requested and returned, the letter map and the bound; usage `prompt_tokens` and `completion_tokens`.
- **The record**: kinds `noul`, `choice`, `score`; text and JSON; `max_questions` 1, so Fit packs one question per wire request; `max_labels`, `max_levels` and `max_total_options` 20; `min_levels` 1; no images: `max_images`, `max_images_per_question`, `max_image_bytes`, `max_total_image_bytes` and `min_image_side` 0, `image_media_types` empty; `max_input_tokens`, `max_question_tokens`, `max_body_bytes` and `truncates_silently` unknown; `returns_refusal`, `calibrated`, `recounts_state_per_question` and `carries_structured_instructions` false; `snapshot_pinnable` and `carries_noul_descriptions` true; no confidence; rounding none.
- **The estimator**: 30 tokens of overhead, the state's text and the question's JSON at 4 characters a token, margin 0.1; `gpt-4.1` counted 58 to 70 prompt tokens on the recorded requests, under it.
- **What the wire did.** `gpt-4.1` reported the dated `gpt-4.1-2025-04-14`, took `max_completion_tokens: 1` and returned 20 top logprobs; 21 were refused with 400.
  Its top-k held the letter, its leading-space and lower-case forms, and also look-alikes in other scripts and prefixed forms such as `(A`, which the protocol does not count.
  On a confident answer those filled the top 20, so a level's letter fell outside it and read as zero mass, with a warning.

## Cloudflare

Two leaves on Workers AI's REST route, `POST https://api.cloudflare.com/client/v4/accounts/<account>/ai/run/<catalog id>` with `Authorization: Bearer <token>`: `Clef(*, model="clef", account_id=None, api_token=None, client=None, capabilities=None)`, named `cloudflare-clef`, for `noul`, `choice` and `score` (`model` is `clef` or `clef-flash`, each on its own route `@cf/cloudflare/<model>`: Clef's route refuses `clef-flash` with 422 and code 3030; each is also taken by that full name, and the leaf reports the short name either way), and `Clairic(*, model="clairic", account_id=None, api_token=None, client=None, capabilities=None)`, named `cloudflare-clairic`, model `clairic`, for `extraction` on `@cf/cloudflare/clairic`.
The account and token default to `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`; without either, or with another model, construction raises `ValueError`.
The client, `capabilities` and `aclose` are as for System One.

`cloudflare(*, account_id=None, api_token=None, clef_model="clef", client=None, retry=RetryPolicy(), concurrency=None, rate=None)`, in `idia.cloudflare` (imported by name, not from the package root), returns a `Router` keyed `noul`, `choice` and `score` to a dispatcher over Clef and `extraction` to one over Clairic.
Both leaves send over one client: with none given, Clef creates it, Clairic shares it, and the router's `aclose` closes it; an injected client is never closed.
Each leaf's `aclose` closes that leaf: after Clairic's own, its calls raise `the leaf is closed` before anything is sent while Clef still sends; after Clef's, Clairic's lender is closed, so it is refused the same way.
The created client follows the running event loop: Clef makes a fresh one on another loop, and Clairic sends with whichever Clef holds, so one router serves successive `asyncio.run` calls over one client per loop.
`retry`, `concurrency` and `rate` go to both dispatchers, the same limiter instances to each.
The credential stop holds within each dispatcher, as [Dispatch](dispatch.md) says: on a token refused with 401 or 403, a request with questions for both leaves makes one call on Clef and one on Clairic, and a request asking only label kinds or only extraction makes one.
A request's options reach every member, so a mixed request carrying `reference_date` hands it to Clef as well, which takes it without a warning (below); any option neither leaf knows draws an `unknown-option` warning on every outcome, as the rule for an unknown option says.

- **The envelope.** Every body comes wrapped in `{success, errors, messages, result}`.
  A success whose `success` is not true, or whose `result` is not an object, is a non-retryable `ProviderError`, as is a `result` without an `answers` object, a non-empty string `model` and a `usage` object.
  `raw_body` is the whole envelope.
- **The status mapper** is the table above, its message the first `errors[]` entry's `code` and `message`; codes 3036 and 4006, each the daily neuron allocation spent, are not retryable on whatever status they arrive, a 429 among them; every other code takes the table's default, so every other 429, 3040 (capacity) among them, is retryable.
  The request id is `cf-ai-req-id`, the id the wire also writes into its error messages.
- **Images** go in a top-level `images` array as `{content_type, base64}`, with the type Fit judged: the declared media type when it is PNG, JPEG or WebP (`image/jpg` read as `image/jpeg`), otherwise the type the bytes' signature shows.
  The records declare the image limits (table below), so Fit refuses before any call more than four images, an image over 4 MiB, images over 8 MiB in all, or another type; a leaf called directly with such an image sends it, with its declared type or else `application/octet-stream`, and the wire refuses it.
  `min_image_side` is unknown, so every outcome of a request whose state has an image carries an `unchecked-limit` warning naming it; `prompt_sent` replaces each `base64` with `bytes`, its size.
- **Question ids.** The wire takes a Clef question id matching `[A-Za-z0-9_.-]{1,100}` and a Clairic field name matching `[A-Za-z_][A-Za-z0-9_.\-]{0,63}`, and refuses others (Clef's `match:intent`, on `clef`, with 422 and code 3030, the cassette `question-id-probe`; Clairic's `1bad` with 422 and code 3030, the cassette `field-name-probe`).
  When every key of a wire request matches, the keys are sent as given; when any does not, every question of that wire request is sent as `q0..qN` (Clef) or `f0..fN` (Clairic), by position, and the answers are mapped back to the caller's keys.
  So `key_patterns` is empty and Fit checks no key.
  `Response.answers` and a `Fault`'s key are always the caller's; an answer under an id not sent is dropped with an `extra-answer-key` warning; `prompt_sent` and a wire error message keep the ids as sent; Clef's estimator counts them as sent.
- **Provenance**: `passthrough`; `calibrated` false; `model_reported` the short name the wire reports, never a snapshot, and every response warns `no-snapshot` that alias drift cannot be detected from it; `usage` the integer `input_tokens` and `output_tokens`, and `passes` where the wire reports it, as the wire reports them, with the whole `usage` object raw: Clef and Clairic report `output_tokens` 0 on every response, the 0 kept as reported, and a figure the wire leaves out is `None` (see [`Usage` in vocabulary.md](vocabulary.md#usage)).

**Clef** sends `model, state, questions[, images]`, compact and the same every time.
The state is built as System One builds it with the images taken out, and is the empty string when only images remain; the questions and the reading per kind are System One's (a `choice`'s probabilities in label order with `chosen` and `confidence`; a `score`'s by index with `expected_level` and the legend echoed as descriptions; a `noul`'s probability).
Clef sends the questions, as Clairic sends its fields, in the wire request's order and never reorders them.
A described `noul`'s `criteria` and a structured instruction are sent as System One sends them.
Through `cloudflare()`, a structured instruction on a label kind reaches Clef and is sent; the router's derived `carries_structured_instructions` is false, since Clairic's is, and a composite's own check does not read it, so each member's record decides.
The wire has no refusal.
No record field holds Clef's 16 MP per image; the wire refuses a larger image.

**Clairic** sends `model[, images][, state], fields[, reference_date]`.
Its constructor takes `model` as `clairic` or `@cf/cloudflare/clairic` and refuses any other; the leaf reports `clairic` either way.
It reads one request option, `reference_date`, an ISO date string that resolves dates written without a year; it is sent and kept in `prompt_sent`, and any other value is refused before sending.
Clef reads no option, and takes Clairic's `reference_date` without an `unknown-option` warning and sends nothing for it, since the `cloudflare` router hands both leaves the same options.
The state is the one text part; several text parts or a JSON part are refused by the leaf with a non-retryable `ProviderError`, one failed attempt with nothing sent, since the spans index the text sent, and a state of images alone sends no `state`.
Each question becomes a field from its value type and instruction alone, `{type, instructions}`, or for a list `{type: "list", instructions, items: <type>}` (the item type as a string; an object there is refused with 400 and code 5006, and the cassette `list-items-object` keeps that refusal as a probe the leaf no longer sends); a list of `time`, `datetime` or `duration` is refused by the leaf with a non-retryable `ProviderError`, one failed attempt with nothing sent, Clairic's lists holding text, numbers or dates.
Nothing else is sent: a unit, a precision or a null threshold is whatever the instruction says and Clairic's default.
A structured instruction, which Clairic's record does not carry and Fit refuses, is refused by the leaf too, for a call made past Fit, before sending with a non-retryable `ProviderError` naming the question, since the type of Clairic's `instructions` is undocumented.

- **Reading**: `value` and `guess` as returned, a date, time or datetime read from its ISO form and a duration's number as seconds when the answer's `unit` is `seconds` or absent, anything that does not read left as given for validation to fault; `confidence` and `p_null` as returned, `None` where absent, as for a list; `span`, or a list's `spans`, as `Span(part=<the text part's index>, start, end)`, none without a text part.
  Clairic, as recorded, gives a span only for a text value or a list of text, so a number, date, time, datetime or duration answer has no span.
  An answer of type `estimate` or `forecast`, or carrying a distribution's keys, is a `Fault` with `Rule.candidate_shape` naming the candidate.
  `truncated: true` adds a `provider-truncated` warning on `state` and the answers stand; `state_text` is ignored.

| Field | Clef (`clef`; `clef-flash`) | Clairic |
|---|---|---|
| `kinds`; `modalities`; `image_carriers` | `noul`, `choice`, `score`; text, JSON, image; `array` | `extraction`; text, image; `array` |
| `max_input_tokens`, `max_question_tokens` | 65,536 for both on `clef`, 24,000 for both on `clef-flash` | 65,536 |
| `max_body_bytes`; `max_questions` | 13,631,488; 64 | 16,777,216; `UNBOUNDED` |
| `max_images`, `max_images_per_question` | 4 | 4 |
| `max_image_bytes`; `max_total_image_bytes` | 4 MiB; 8 MiB | 4 MiB; 8 MiB |
| `image_media_types`; `min_image_side` | PNG, JPEG, WebP; unknown | PNG, JPEG, WebP; unknown |
| `max_labels`, `max_levels`, `min_levels`; `max_total_options` | 255, 10, 2; `UNBOUNDED` | unknown |
| `key_patterns` | none | none |
| `truncates_silently` | true | false |
| `confidence`; `rounding` | `undocumented` for `choice` and `score`; four decimals, as recorded | `undocumented`; none |
| `calibrated`, `snapshot_pinnable`, `returns_refusal`, `recounts_state_per_question` | false | false |
| `carries_noul_descriptions`, `carries_structured_instructions` | true | false |

**The estimators**, margin 0.1, fitted to the recorded usage: Clef's is 260 tokens of overhead, the state's JSON at 4 characters a token, the questions' at 4, and 1,024 tokens per image, the cap of the image formula, since an image part carries no size; Clairic's estimates one pass holding every field, which is what the context limit bounds: the state's text at 3 characters a token, 150 tokens a field and 1,024 per image.

**What the wire did.** Clef and clef-flash, each on its own route, reported their short names and four-decimal probabilities, confidences and expected levels, `output_tokens` always 0, and counted a long state once whether asked one question or three (1,625 and 1,791 input tokens); 65 questions were refused with 422 and code 3030, and Fit packs them in two wire requests first.
Clairic's spans are code points: a target after characters outside the BMP sliced exactly.
It answered `time`, `datetime` and `date` fields as ISO strings (a date written without a year resolved by `reference_date`), a duration as a number with `unit: "seconds"`, a list with `value`, `spans` and `confidence` and no `p_null` or `guess`, a number with two decimals as returned (`4250.5`), and a text field it could not settle with a null `value`, a `guess` and its span; it reported `output_tokens` 0 and split two fields into two passes, each counting the state again in `input_tokens`, the count of passes reported in `usage` as `passes`.
A field asked as `estimate` came back with `mean`, `median`, `p10`, `p90`, `p_below` and `p_above`, which the leaf faults as `candidate_shape`; an invalid field name was refused with 422 and code 3030.
Each model answered a request whose keys no wire pattern takes (`q:match/<key>`), sent under `q0, q1` or `f0, f1` (the `alias` cassettes), with the same labels and values as the same request under plain keys; the probabilities moved by up to 0.009 and the confidences by up to 0.022 on Clef and clef-flash, and Clairic's unsettled field's `p_null` by 0.10 and `confidence` by 0.13.
An identical body sent twice answered identically on clef, clef-flash and Clairic, and reversing four questions moved a probability by up to 0.053 on clef and 0.032 on clef-flash, and a confidence by up to 0.143 and 0.085, while reversing Clairic's two fields moved nothing.
Adding to a request one question whose key sorts first, so that each of its other questions moved one place, moved every other question's probability by up to 0.13 on Clef and none on `openai-decisions`; a neighbour can move a Clef answer by up to 0.13, more than reordering did.
Asked alone and asked first of three, the same question moved by 0.010 on clef and 0.024 on clef-flash; so the alias and neighbour moves above come from the change to the body, not from run-to-run variation, unless the model behind the short name changed between recordings.
