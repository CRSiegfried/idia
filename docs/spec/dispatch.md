# Dispatch: the deadline, retries, limits and outcomes

A leaf answers one wire request in one call and is not a provider until wrapped.
`Dispatcher(leaf, *, retry=RetryPolicy(), concurrency=None, rate=None, estimator=None, overflow=Fail(), reask=False)` wraps it into a `Provider`; its `capabilities` are the leaf's.
`reask` and `overflow` are read-only properties, what the dispatcher was built with.
Given a `Dispatcher`, `Router` or `Fallback`, or anything that is not a `Leaf`, it raises `TypeError` at build, naming the type.
`idia.provider(name, *, retry, concurrency, rate, **options)` builds the leaf registered as `name` and returns it wrapped the same way, typed as the `Dispatcher` it is: `retry`, `concurrency` and `rate` go to the dispatcher and every other keyword to the leaf.
Each registered name has its own overload, typed with its leaf's keywords, and a last overload takes any `str` with any keywords; a registered name given a keyword its leaf does not take, a value of the wrong type, or without a keyword its leaf requires, matches a deprecated overload instead, which a type checker reports as deprecated, an error under pyright strict; a keyword the leaf does not take raises `TypeError` naming the call and the name, as `provider('openai-decisions'): ` followed by the leaf's message, with the leaf's error as its cause; for `estimator`, `overflow` or `reask` the message adds that they are the `Dispatcher`'s and the leaf is to be wrapped by hand.
`idia.leaf(name, **options)` builds the leaf `idia.provider(name, **options)` would wrap, through the same guard, and returns it unwrapped; its overloads are `provider`'s less `retry`, `concurrency` and `rate`, each typed to return `Leaf`, and a keyword the leaf does not take raises `TypeError` named `leaf('<name>'): `, which for `retry`, `concurrency`, `rate`, `estimator`, `overflow` or `reask` adds that they are the `Dispatcher`'s.
To set `estimator`, `overflow` or `reask`, build the leaf, by name with `idia.leaf` or from its class, and wrap it in a `Dispatcher` by hand:

```python
import idia
from idia import Dispatcher

decisions = Dispatcher(idia.leaf("openai-decisions"), reask=True)
```

A caller may wrap a leaf of its own around the one it built, such as one counting every call for a budget: each attempt, retries included, is one `call` on it, since the loop alone retries.
Such a wrapper forwards `name`, `model`, `capabilities` and `estimator`, which the `Dispatcher` reads when it is built, and an `aclose` to the inner leaf's, which `Leaf` does not require but without which closing the `Dispatcher` leaves the inner leaf open.

An unknown name raises `ProviderNotFound`, whose `name` is the name asked for and whose `registered` holds the registered names, sorted, which its message lists.
`idia.provider_names()` returns the same tuple, listing a name whether or not its extra is installed.

`Dispatcher`, `Router` and `Fallback` each have `async aclose()` and are async context managers: a dispatcher calls its leaf's `aclose()` when the leaf has one and leaves a leaf without one alone, and a composite closes each distinct member once, in member order, trying every member and re-raising the first error after all were tried.
Closing is how a leaf built through `idia.provider` releases the HTTP client it created; `Leaf` does not require `aclose`, and a leaf's `aclose` is awaited, so it must be a coroutine function.
Closing again calls the leaf's `aclose` again; Dispatch keeps no closed flag, so a leaf's `aclose` must tolerate repeats.

## One deadline scope

On `answer`, the dispatcher first runs Fit's preflight and plan against the leaf's record, with the leaf's estimator unless another was given; `DoesNotFit` is raised before any call.
`plan(request)` runs that preflight and plan alone and makes no call: it uses the leaf's record and `model`, the leaf's estimator unless another was given, and the dispatcher's overflow policy, and returns the [`FitPlan`](fit.md#the-plan) that `answer` runs, or raises `DoesNotFit`.
Then, for each wire request in plan order, in sequence:

- If the deadline has already passed, as `Deadline.passed()` reads it, the wire request is never sent: its outcome fails with `DeadlineExceeded` and lists no attempts.
- Otherwise the attempt loop: take the concurrency slot, then a rate token, then call the leaf with the wire request and the caller's deadline, which is the leaf's own timeout.
  The slot is held through the rate wait and the call, and given back before a retry's sleep.
- A `ProviderError` from the call is a failed attempt.
  The attempt is retried when the error's `is_retryable` is true, the policy has attempts left, a limiter can be taken before the deadline, and the retry delay ends before the deadline.
  When the error is not retryable or the policy's attempts are spent, the wire request fails with that error, the last one raised.
  When a retry is due but the deadline cannot fit it, its delay ending past the deadline, a limiter refusing or its wait ending past the deadline, or the deadline passing before the call, the wire request fails with `DeadlineExceeded`, saying the deadline ended it; the attempts made stay listed with the provider's errors, and the outcome's own `request_id` is `None`, each attempt keeping its own.
- A wire request that fails with a credential error, a `ProviderError` whose `is_retryable` is false and whose `status_code` is 401 or 403, leaves every later wire request of the call unsent.
  Each unsent one has an outcome as for a wire request never sent because the deadline passed, but with the credential error itself, the same object: its keys, the leaf's name and model, no attempts, no request id, its own `started_at`, a near-zero `latency` and preflight's `unchecked-limit` warnings for it.
  An unsent outcome holds the same error object as the failed one, so `outcome.error.request_id` names the failed wire request while the unsent outcome's own `request_id` is `None`.
  The stop belongs to that `answer` call alone: the dispatcher keeps no flag, so a concurrent or later call sends as usual.
  It also ends at that dispatcher: a composite's other members are not stopped by it.

Every limiter wait, retry sleep and call counts against the one deadline the caller named.
Nothing restarts it.
The rate limiter, whose wait is known in advance, raises `DeadlineExceeded` at once when the wait would end after the deadline, waiting for nothing; the concurrency limit waits for a slot until the deadline and then raises it.
The dispatcher checks the deadline once more after the limiters are taken for every attempt, since a wait can end late and a call cut at the deadline can end up to one clock tick before it; this check, like the one before the first send, asks `Deadline.passed()`.
A wait that ends past the deadline is a refusal too, and the slot is given back, but a token taken by a wait that ends late is not returned.
Either way: before the first attempt, the wire request is never sent; before a retry, it fails with `DeadlineExceeded`, its failed attempts listed.

## The retry policy and the limits

`RetryPolicy(attempts=3, base_delay=0.5, jitter=0.25, honour_retry_after=True)`.
`attempts` counts calls, so `attempts=1` never retries.
The delay after `n` failures is the error's `retry_after` when it has one and the policy honours it, used as given; otherwise `base_delay * 2 ** (n - 1)`, spread by up to `jitter` either way.
A fault in an answer is not a `ProviderError` and never reaches the policy.

`ConcurrencyLimit(limit)` allows `limit` calls in flight at once.
`RateLimiter(rate, per=1.0)` is a token bucket of `rate` calls per `per` seconds, holding at most `rate` tokens and starting full, one token per attempt; with a `rate` below 1 the bucket never holds a whole token, so even the first attempt waits for one, whereas `RateLimiter(1, 2.0)` is one call every two seconds with no first wait.
One instance may be shared by two dispatchers whose leaves share one vendor limit.
A `ConcurrencyLimit` follows the running event loop: its semaphore is made for the loop it is first used on and made again on another, so one `Dispatcher` serves successive `asyncio.run` calls, the limit holding within each loop.

One `Dispatcher` may serve several `answer` calls at once on one event loop.
Each call has its own plan, deadline scope, retries, failed attempts and outcomes; the dispatcher keeps no state of its own that a call changes.
The calls share its leaf, its `ConcurrencyLimit` and its `RateLimiter`: at most `limit` calls are in flight across all of them, and every attempt of every call takes one token from the one bucket.
The leaf's `call` is then awaited concurrently, so the leaf must allow that.
`aclose` must not be awaited while a call is in flight.

## The outcome

One outcome per wire request, in plan order.
It carries the keys the wire request held, the leaf's `name` and `model`, and the wire response's provenance as the leaf gave it: the model reported, the snapshot, the elicitation, `calibrated`, the prompt sent, usage and the request id.
The raw body goes no further than the wire response.
The dispatcher adds `started_at` (wall clock), `latency` (monotonic, over the whole wire request, waits and retries included) and `failed_attempts`, one per `ProviderError` with the leaf's name and model, the error, its request id, the attempt's start and duration.
`attempts` is derived: the failed attempts plus one when the request answered.
A failed outcome has the error, no prompt sent and no usage.

Warnings: an outcome's are the wire response's followed by preflight's `unchecked-limit` warnings for that wire request, on a failed outcome too; the response's are the warnings about the request as a whole rather than one wire request, which in this release is preflight's `overflow-policy` warning alone.

## Three endings

- `DoesNotFit` before any call.
- A `Response` when at least one wire request answered.
  Its answers are the plan's merge: a failed wire request's keys are absent.
  An outcome whose answers are all faults still answered.
- `RequestFailed` carrying the full response when preflight passed and no wire request answered, whether calls were made or not.

## The re-ask

With `reask=True`, each plan wire request whose answers hold a `Fault` is sent once more, with the same state and only the faulted questions, through the same loop under the same deadline.
A `Refusal` is not re-asked.
The re-ask's outcome is appended after the plan's outcomes, and only the keys its answers hold override the first answers.
A re-ask that fails leaves the plan's fault standing in `answers`, with the failed re-ask outcome beside the plan's.
After a credential error, in the plan or in an earlier re-ask, a re-ask is left unsent in the same way: its unsent outcome is appended after the plan's, and the plan's fault stands.

## Router and Fallback

Composites are built from idia's own providers, a `Dispatcher`, `Router` or `Fallback`, never from leaves and never from another provider, which is refused at build with `TypeError`; they satisfy `Provider`.
A composite has no `plan`; its members' checks run inside `answer`.

`Router(members: Mapping[Kind, Provider])` refuses at build a kind routed to a member whose record lacks it, with a `ValueError` naming the member, a dispatcher by its leaf's name and model, and the kinds its record has.
An empty `members` raises `ValueError` at build, as an empty `Fallback` does.
A mapping held in a variable is annotated, as `routes: dict[Kind, Provider] = {"noul": decisions, "extraction": clairic}`, because a type checker infers an unannotated dict literal's keys as `str` and its values as the members' own type, and `Mapping` is invariant in its key, so a strict checker refuses `Router(routes)`; a literal written in the call itself needs no annotation.
Each member gets the questions of its kinds as one request of its own, with the request's state, deadline and options unchanged, the members in the mapping's order of first appearance; a member with no questions is not called.
Members run concurrently, so a credential error stops only the plan of the dispatcher it came from: every other member with questions still makes its first call.
The outcomes are concatenated in member order, the answers are the union, and the response warnings are the members' concatenated.
A member's `RequestFailed` joins the others' responses; when every member fails, `RequestFailed` carries them all.

`Fallback(members: Sequence[Provider])` gives each member in turn the same request and the same deadline; an empty `members` raises `ValueError`.
It falls through on `RequestFailed` only; a `DoesNotFit` propagates.
A member stopped by a credential error before any of its wire requests answered raises `RequestFailed`, so the next member is tried, by design, since it may hold a different credential.
A member that answered part of its plan before the error returns its `Response`, and no later member is tried.
The failed members' `failed_attempts` are listed, in order, before the answering member's own on its first outcome, whether that outcome answered or failed; that outcome's `latency` is its own plus the time the earlier members took, and its `started_at` is the fallback's own start.
The answering member's other outcomes keep their own `latency`, `started_at` and `failed_attempts`.
When every member fails, `RequestFailed` carries every member's outcomes in member order, with no answers and the members' response warnings concatenated.

**Before any call**, a composite checks the request against its derived record for kinds and modalities, then each member runs its own full preflight and plan on its share.
The `unchecked-limit` warnings come from the member checks.

**The derived record** is built once, when the composite is.
A router's kinds are the mapping's keys and its modalities the union; a fallback's are the intersections.
Every other field takes the most restrictive value: a limit is the smallest known number, else `None` when any member's is unknown, else `UNBOUNDED`; `min_levels` the largest known; `calibrated` and `snapshot_pinnable` hold only when every member's does; `returns_refusal` and `recounts_state_per_question` when any does; `truncates_silently` is true if any member's is, else `None` if any is undocumented, else false; `rounding` takes the fewest decimals, a known number beating `None`; `key_patterns` are concatenated.
`carries_noul_descriptions` holds only when every member that answers a `noul` declares it, a router's from the members that route `noul`, and is false when none does; `carries_structured_instructions` holds only when every member's does.
A router's `max_labels`, `max_levels`, `min_levels`, `max_total_options` and confidence map come from the members that route those kinds; a fallback's confidence map drops a kind any member lacks and names differing formulas `mixed`.
`image_carriers` are the union when the derived modalities include `image`, otherwise empty.
The four image fields come from the members that take images alone: `max_image_bytes` and `max_total_image_bytes` are the most restrictive of their values, as any limit is, `min_image_side` the largest known, and `image_media_types` the intersection of the known sets, which may be empty; each is `None` when no such member knows it or no member takes images.
The composite's own check reads none of the four: each member's check does.
A composite has no opt-in features, and its `feature_kinds` and `feature_modalities` are empty; its members' records keep theirs.
The record is marked `composite`, so the leaf-only refusal of silent truncation with an unknown token limit does not apply.
