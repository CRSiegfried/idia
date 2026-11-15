# Ask a provider

In this tutorial you put questions to Cloudflare's decision models through idia: a yes/no
question and a choice to Clef, an extraction to Clairic, both kinds at once through the
`cloudflare()` router, and finally a request that fails, to see what idia hands back. You
need Python 3.11 or later and a Cloudflare account id and API token with Workers AI access.
Every example makes a live call. The [README](../../README.md#try-it-without-keys) shows the
same shapes with no keys, against the scripted leaf.

## Install

Install idia with its `cloudflare` extra, and export the two credentials it reads; idia reads
no `.env` file:

```sh
pip install "idia[cloudflare] @ git+https://github.com/CRSiegfried/idia@v0.2.0"
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
```

## A yes/no question and a choice, to Clef

A request holds one input, the state, and the questions about it. Each question has a key,
which names its answer, and a kind. Put a `noul` and a `choice` to Clef in one request:

```python
import asyncio

import idia
from idia import Deadline, Label, Question, Request, State, TextPart


async def main() -> None:
    request = Request(
        state=State(parts=(TextPart(text="The invoice is overdue and I want a refund."),)),
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
    async with idia.provider("cloudflare-clef") as clef:
        response = await clef.answer(request)
    for key, answer in response.answers.items():
        print(key, answer)


asyncio.run(main())
```

`complaint` comes back as a `NoulAnswer`, the probability of yes, and `topic` as a
`LabelAnswer`, a probability for each label. Clef does not claim its probabilities
calibrated: each outcome's `calibrated` is `False`.

## An extraction, to Clairic

An extraction asks for a value of a declared type found in the input. Clairic answers it:

```python
import asyncio

import idia
from idia import Deadline, Question, Request, State, TextPart


async def main() -> None:
    request = Request(
        state=State(parts=(TextPart(text="Please ship by 14 March 2026 to Lyon."),)),
        questions=(
            Question(
                key="city", kind="extraction", instruction="The delivery city.", value_type="text"
            ),
        ),
        deadline=Deadline.after(30),
    )
    async with idia.provider("cloudflare-clairic") as clairic:
        response = await clairic.answer(request)
    print(response.answers["city"])


asyncio.run(main())
```

## Both kinds at once, through the router

`cloudflare()`, from the `idia.cloudflare` module, is one provider for both: a router that
sends the label kinds to Clef and extractions to Clairic, each its share of the request. The
response holds the answers from both, and one outcome for each wire request, Clef's first,
each naming the leaf that took it; if one member fails, its keys are missing from the answers
and its outcome says why:

```python
import asyncio

from idia import Deadline, Question, Request, State, TextPart
from idia.cloudflare import cloudflare


async def main() -> None:
    request = Request(
        state=State(parts=(TextPart(text="Please ship by 14 March 2026 to Lyon."),)),
        questions=(
            Question(key="urgent", kind="noul", instruction="Is the sender in a hurry?"),
            Question(
                key="city", kind="extraction", instruction="The delivery city.", value_type="text"
            ),
        ),
        deadline=Deadline.after(30),
    )
    async with cloudflare() as router:
        response = await router.answer(request)
    for key, answer in response.answers.items():
        print(key, answer)
    for outcome in response.outcomes:
        print(outcome.provider, outcome.keys)


asyncio.run(main())
```

## When nothing answers

Each answer is a `NoulAnswer`, `LabelAnswer`, `ExtractionAnswer`, `BoxAnswer`, `Refusal` or
`Fault`; a fault is scoped to its question, and its siblings stand. A request idia cannot
split to fit (a kind the provider lacks, too many labels on one question, an input too long
for one call, a wording form its record does not carry) raises `DoesNotFit` before any call;
it is never truncated. When no wire request answers, `answer` raises `RequestFailed`, which
carries the response and its outcomes. Give Clef a token that is not one and read why:

```python
import asyncio

import idia
from idia import Deadline, Question, Request, RequestFailed, State, TextPart


async def main() -> None:
    request = Request(
        state=State(parts=(TextPart(text="Hello"),)),
        questions=(Question(key="greeting", kind="noul", instruction="Is this a greeting?"),),
        deadline=Deadline.after(30),
    )
    async with idia.provider("cloudflare-clef", api_token="not-a-token") as clef:
        try:
            await clef.answer(request)
        except RequestFailed as failed:
            print(failed)
            for outcome in failed.response.outcomes:
                print(outcome.provider, outcome.attempts, outcome.error)


asyncio.run(main())
```

The outcome names the leaf, how many calls it made, and the error that ended it; the
exception's message names the same.

## Where next

- [The providers spec](../spec/providers.md): what each provider takes, its limits and its
  models.
- [Dispatch](../spec/dispatch.md): the deadline, retries, rate limits, the router and the
  fallback.
- ommat's tutorial [Decide with idia](https://github.com/CRSiegfried/ommat/blob/main/docs/tutorials/decide-with-idia.md)
  binds these providers as oracles.
