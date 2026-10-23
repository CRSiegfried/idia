# Fit: preflight, packing and overflow

Before any call, idia checks a request against the provider's `Capabilities` record and plans the wire requests it becomes.
A request that cannot be sent raises `DoesNotFit` before any call.
`check` names the failing field, such as `max_input_tokens`, `kinds` or `key_patterns`.
`observed` is what the request had and `limit` what the field allows.
`state_overflow` is true when the state alone does not fit.
`off_features` lists opt-in features the provider offers that this instance has not turned on.
For a missing kind or modality it lists only those the record's `feature_kinds` or `feature_modalities` says add a missing value; for any other check it lists every one, since no record says what a feature does to a limit.

## The message for a missing kind or modality

The message is `<field> not supported: <missing values, sorted, joined by ", ">`, then its clauses, joined by "; ".
For a leaf, each missing value some off feature adds has one clause, naming the features sorted and joined with "and", with no comma before it: "the images feature adds it", "the a and b features add it", or "the a, b and c features add it".
The values no feature adds share one clause, placed where the first of them falls, which names the model: "no feature of model jev-1.13.0 adds it", or "no feature of this model adds it" when preflight was given no model name; the Dispatcher always gives its leaf's.
With one missing value the clause says "it"; with several, each clause names its values, those no feature adds joined with "or": "adds box or extraction", or "adds a, b or c".
A composite's own check has one clause whatever the count: "no feature of this composite adds it; its members' own records may", with "them" for several values.
A router's own check for kinds, whose kinds are those it routes, reads instead "this router routes it to no member", or "routes them"; its modality check, and a fallback's two checks, keep the composite clause.

| Case | Message |
|---|---|
| One feature adds it | `kinds not supported: box; the images feature adds it` |
| No feature of the model adds it | `kinds not supported: box; no feature of model jev-1.13.0 adds it` |
| A modality | `modalities not supported: image; the images feature adds it` |
| Several values | `kinds not supported: box, extraction; the images feature adds box; no feature of model jev-1.13.0 adds extraction` |
| Several values no feature adds | `kinds not supported: box, extraction; no feature of model jev-1.13.0 adds box or extraction` |

## Limits

A limit that is a number is checked.
`UNBOUNDED` is neither checked nor warned.
`None` (unknown) is skipped, and each wire request it bears on carries an `unchecked-limit` warning naming the field.
A limit bears on a wire request only where it has something to count: `max_labels` where the request carries a `choice`, `max_levels` and `min_levels` where it carries a `score`, the image limits where the state has an image, and `max_total_options` where the option count is above zero.
The token, body and question-count limits bear on every wire request.
`min_image_side` 0 is a documented absence of any minimum: it is neither checked nor warned.
An image whose size cannot be read is not checked against a known `min_image_side`; each wire request then carries one `unchecked-limit` warning with field `min_image_side` naming every such image, as `the sizes of images 0 and 2 could not be read, so min_image_side was not checked`.
Images are counted from 0 among the state's images, in state order.

## The checks

Each check runs against the leaf's record:

- Every question's kind is in `kinds`, and every state part's modality (text, JSON or image) is in `modalities`.
  `observed` is the missing kinds, or the missing modalities, not every one asked, and `limit` the record's `kinds` or `modalities`.
- A state with an image needs a non-empty `image_carriers` (check `image_carriers`).
  The image count is at most `max_images`, and at most `max_images_per_question`, since every question sees every image.
- Each image, where the state has one, against the four image fields, in this order:
  its media type is in `image_media_types`, read as its declared `media_type` lower-cased with `image/jpg` read as `image/jpeg` when that is in the set, else as the type its signature shows (PNG, JPEG or WebP) when that is (check `image_media_types`, `observed` the declared type, else the signature's, else `"unknown"`, and `limit` the set);
  its raw bytes, not their encoding, are at most `max_image_bytes`, and all the images' raw bytes together at most `max_total_image_bytes`;
  and its shorter side, read from its PNG or JPEG header, is at least `min_image_side` (`observed` the shorter side).
  A failure on any of the four names every off feature.
- Every key matches every pattern in `key_patterns` in full (`re.fullmatch`).
  Every failing key is named, in question order, each with the first pattern it fails; `observed` is the tuple of failing keys and `limit` the record's `key_patterns`.
- A `choice` has at most `max_labels` labels.
  A `score` has at most `max_levels` levels and at least `min_levels`.
- With the state, each question is at most `max_question_tokens` by the estimate.
- Each wire request is within `max_questions`, `max_total_options`, `max_body_bytes` and `max_input_tokens`.

**Tokens.** The estimator is the leaf's own unless the dispatcher names another.
An estimate `TokenEstimate(tokens, margin)` counts as `ceil(tokens * (1 + margin))`, with the margin read as the decimal it prints as.
A wire request counts `estimate(state, questions)`.
By choice, where the record has `recounts_state_per_question`, a wire request of `n` questions adds `(n - 1)` times the state's own estimate, `estimate(state, ())`.
Each estimate takes its margin before the sum.

**Options.** A wire request's option count is the sum over its questions: a `choice`'s labels, a `score`'s levels, 2 for a `noul`, and 0 for an `extraction` or a `box`.

**Body size.** Fit measures the body itself.
The state's text and JSON parts count as compact UTF-8 and its images as the length of their base64 encoding.
Each question adds its key, instruction, label names and descriptions in UTF-8.
By choice, the wire envelope a leaf adds is not counted, so `max_body_bytes` holds only approximately: a leaf's serialized body can be somewhat larger than Fit's measure.

## Packing

A request asks at least one question.
The whole request is checked first: if it fits, it is one wire request carrying the state and every question, whatever the estimator says of smaller sets.
Only when it does not fit are the questions packed, greedily and in order.
A wire request takes the next question while `max_questions`, `max_total_options`, `max_body_bytes` and `max_input_tokens` all hold.
Otherwise a new wire request starts.
Each candidate is estimated whole, `estimate(state, chunk)`, never as a sum over its questions.
Every wire request repeats the state, and every key lands in exactly one, in question order.
A question that does not fit alone with the state raises `DoesNotFit` naming the field.
The plan is deterministic, the same questions in the same order always packing the same way, but which questions share a wire request depends on their order and on where a limit splits them, so adding, removing or reordering a question can move others into a different wire request, and on a provider whose answers depend on the other questions sent with them, change those answers.

## State overflow

The state overflows when its own estimate exceeds `max_input_tokens` or `max_question_tokens`, or its own bytes exceed `max_body_bytes`.
A token overflow goes to the overflow policy, `apply(state, limit, estimator)`, where `limit` is the tighter limit the state is over.
The default, `Fail`, raises `DoesNotFit` with `state_overflow` true.
A named policy returns a new state, which is checked again from the state checks on.
If it still overflows, `DoesNotFit` is raised; the policy is applied once.
The response then carries an `overflow-policy` warning whose `field` is the limit the state was over and whose message reads "the state overflowed <field>; policy '<name>' applied".
By choice, a body overflow of the state always raises, whatever the policy: a named policy applies to token overflow only.
Nothing is truncated silently.

## The plan

Preflight and packing return a `FitPlan`, exported from `idia` and returned by [`Dispatcher.plan`](dispatch.md#one-deadline-scope).
It holds `wire_requests` in plan order; `outcome_warnings`, parallel to them, each entry the `unchecked-limit` warnings for that wire request's outcome; and `response_warnings`, the `overflow-policy` warning when a policy was applied.
A `FitPlan` whose `outcome_warnings` is not as long as its `wire_requests` raises `ValueError`.

## Composites

A composite's derived record is checked for kinds and modalities only, with no warnings and no plan.
Every other field, the question count among them, is checked by each member against its own record for its share.

## The merge

The answers are keyed as the questions were, and a failed wire request's keys are absent.
The outcomes are in plan order.
