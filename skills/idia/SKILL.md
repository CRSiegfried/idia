---
name: idia
description: Use when code asks an LLM or decision model closed questions (yes/no, a label, a score, an extracted value, an image region) to classify, route or triage an input and needs the model's probabilities rather than text to parse, or batches many questions about one input into few calls, with the idia Python library over System One, OpenAI (Decisions or chat logprobs), Cloudflare (Clef, Clairic), or GLiNER2 run in process.
---

# idia

idia puts closed questions about one input to a decision-model provider and returns each
answer as the provider gave it (probabilities for yes/no, labels and levels), with who
answered. The questions of one request are packed into calls as full as the provider's
limits allow, each call sending the input once.

## Install

```sh
pip install "idia[system-one] @ git+https://github.com/CRSiegfried/idia@v0.1.0"
pip install "idia[openai] @ git+https://github.com/CRSiegfried/idia@v0.1.0"
pip install "idia[cloudflare] @ git+https://github.com/CRSiegfried/idia@v0.1.0"
pip install "idia[gliner] @ git+https://github.com/CRSiegfried/idia@v0.1.0"
```

Credentials come from the environment unless passed to `idia.provider`: `TYPESAFE_API_KEY`
(System One), `OPENAI_API_KEY`, or `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
idia reads no `.env` file.

## The core pattern

```python
import idia
from idia import Deadline, Label, NoulAnswer, Provider, Question, Request, State, TextPart


async def triage(provider: Provider, message: str) -> float | None:
    request = Request(
        state=State(parts=(TextPart(text=message),)),
        questions=(
            Question(key="complaint", kind="noul", instruction="Is this a complaint?"),
            Question(
                key="topic",
                kind="choice",
                instruction="What is it about?",
                labels=(Label(name="billing"), Label(name="shipping"), Label(name="other")),
            ),
        ),
        deadline=Deadline.after(30),
    )
    response = await provider.answer(request)
    for outcome in response.outcomes:
        print(outcome.provider, outcome.model_reported, outcome.keys, outcome.calibrated)
    answer = response.answers.get("complaint")
    return answer.probability if isinstance(answer, NoulAnswer) else None


async def main(messages: list[str]) -> None:
    async with idia.provider("cloudflare-clef") as provider:  # one provider, many requests
        for message in messages:
            print(await triage(provider, message))
```

- Put every question about one input in one `Request`. That is what batches them; a request
  per question sends the input once per question.
- `Deadline.after(seconds)` takes seconds from now. The deadline is one instant for the
  whole request: every call, retry and rate-limit wait, across all its calls.
- Use `async with`, which closes the HTTP client the provider created; a `client=` you pass
  in stays yours to close.
- Read `outcome.calibrated` before treating a probability as calibrated.
- To reproduce a wording measured elsewhere, a `noul` may carry `Label(name="yes", ...)`
  then `Label(name="no", ...)` descriptions, and an instruction may be a JSON object or a
  tuple. A provider whose `capabilities.carries_noul_descriptions` or
  `carries_structured_instructions` is false raises `DoesNotFit` before any call, so check
  those fields first to leave such a question out (see the
  [providers spec](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/spec/providers.md)).

## Handling results

- `Refusal` in `answers`: the provider declined that question.
- `Fault` in `answers`: that question's answer was missing, unreadable or broke a rule; the
  other answers stand.
- A `chosen-below-max` warning on an outcome: that question's chosen label, kept as the
  provider gave it, is one rounding step below the highest probability.
- A key absent from `answers`: the call carrying it failed, and its outcome says why.
- `DoesNotFit`, raised before any call: the request does not fit the provider's declared
  capabilities (a kind it lacks, a limit no split can meet, or a wording form it does not
  carry).
- `RequestFailed`, raised when no call answered, sent or not: its `response` carries every
  outcome, each with its `error`.

The answer shapes are in the vocabulary spec's
[Answers](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/spec/vocabulary.md#answers)
and the errors under its
[Errors](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/spec/vocabulary.md#errors),
and when each ending happens is in the
[dispatch spec](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/spec/dispatch.md#three-endings).

## Testing offline

`Scripted`, from `idia.providers.scripted`, answers from a script with no keys. Give it a
`Capabilities` record written out in full and wrap it in a `Dispatcher`; the
[README demo](https://github.com/CRSiegfried/idia/blob/v0.1.0/README.md#try-it-without-keys)
is a complete example, and the
[test-doubles spec](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/spec/test-doubles.md#the-scripted-leaf)
covers `Scripted` and `Recorded`. The scripted leaf logs each call in `calls`, and raises
`ScriptExhausted` when its script runs out.

## Pitfalls

- `openai-chat` needs `model=`, a chat model that returns logprobs, and takes one question
  per call, so it does not batch.
- A missing extra makes `idia.provider` raise `ModuleNotFoundError` naming the extra it
  needs; install it with the matching line above.
- A kind the provider lacks raises `DoesNotFit`: Clef takes no `extraction`, Clairic only
  `extraction`, and `gliner` only `extraction` of text. `cloudflare()`, in `idia.cloudflare`,
  routes each kind to the right one.
- `gliner` runs a model in process and needs no credential: build it with
  `idia.leaf("gliner")`, `await leaf.load()` once, then pass it to `Dispatcher(...)`;
  `idia.provider("gliner")` gives a provider whose leaf is never loaded, so every call
  fails. A text past 2,048 tokens is `DoesNotFit`, so split it first.
- A leaf is not a provider: build one with `idia.provider(name)`, or build the leaf with
  `idia.leaf(name)`, wrap it in one of your own if you like (to count every call, say), and
  pass it to `Dispatcher(...)`. `idia.provider_names()` lists the names.
- Only `idia.__all__` and the `__all__` of the modules CONTRIBUTING lists are public; any
  other name may change without notice.

## Details

- [Providers](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/spec/providers.md):
  each provider's options, limits and models.
- [Dispatch](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/spec/dispatch.md):
  the deadline, retries, rate limits, outcomes, the re-ask, `Router` and `Fallback`.
- [Tutorial](https://github.com/CRSiegfried/idia/blob/v0.1.0/docs/tutorials/ask-a-provider.md):
  live calls to Clef and Clairic, the router, and handling `RequestFailed`.
- [README demo](https://github.com/CRSiegfried/idia/blob/v0.1.0/README.md#try-it-without-keys):
  a runnable example with no keys.
- [Public API](https://github.com/CRSiegfried/idia/blob/v0.1.0/CONTRIBUTING.md#ground-rules-for-changes):
  the modules whose `__all__` is public.
