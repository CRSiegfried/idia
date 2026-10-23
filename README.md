# idia

A standalone Python library for putting type-1 questions to decision-model providers.
A type-1 question is one with a fixed answer shape (yes/no, a label set, an ordered scale,
an extracted value, a region over an image), the closed kind of judgment a classifier
makes, as against a type-2 question, which asks for free-form reasoning, text or
structured output. A provider answers one with probabilities over that shape and says
whether it claims them calibrated. idia batches questions over one input, handles
limits, retries and rate limits, and returns the answers with what was sent and who
answered.

**The name.** Idia and Ommat come from *ommatidia*, the thousands of small hexagonal eyes
that make up a compound eye and combine into one picture: *ommat-* is the Greek stem for
"eye", *-idia* "little ones". Many small, cheap questions add up to one full picture.
Idia puts them to the models; Ommat is the engine that asks them.

```text
       ______            _____            ______
      /      \          /     \          /      \
     /  .  .  \        / .   . \        /  . .   \
     \        /        \       /        \        /
      \______/          \_____/          \______/


             ______              ______
            /      \            /      \
           / .    . \          /  .  .  \
           \        /          \        /
            \______/            \______/
```

The core installs with no dependency. Each wire provider is an optional extra.

## Kinds and providers

A question asks for one of five kinds of answer:

| Kind | Answer | What it asks |
|---|---|---|
| `noul` | `NoulAnswer` | Yes or no, answered with one probability, that of yes |
| `choice` | `LabelAnswer` | One label of a set: a probability for each label |
| `score` | `LabelAnswer` | A level on an ordered scale: a probability for each level, with the expected level where the provider gives it |
| `extraction` | `ExtractionAnswer` | A value, or a list of values, of a declared type found in the input, or none, with its span (one per item) where the provider gives it |
| `box` | `BoxAnswer` | A region of an image, as fractions of its sides |

Any question may instead come back as a `Refusal`, where the provider declined it, or a
`Fault`, where its answer was missing, could not be read, or broke a rule; or with no answer,
when the wire request carrying it failed, its outcome saying why.

Each provider is registered under a name that `idia.provider(name)` builds, from the
extra that installs it:

| Name | Extra | Kinds | Claims calibrated |
|---|---|---|---|
| `system-one` | `system-one` | `noul`, `choice`, `score`; with `images=True`, `box` too on `jev-1.14.0` and `jev-preview` | yes |
| `openai-decisions` | `openai` | `noul`, `choice`, `score` | no |
| `openai-chat` | `openai` | `noul`, `choice`, `score` | no |
| `cloudflare-clef` | `cloudflare` | `noul`, `choice`, `score` | no |
| `cloudflare-clairic` | `cloudflare` | `extraction` | no |

`openai-chat` needs `model=`, a chat model that returns logprobs, and `cloudflare-clef` also
takes `model="clef-flash"`. `cloudflare()`, in `idia.cloudflare`,
is a router over Clef and Clairic, shown below. What each provider takes, its limits and its
models are in [the providers spec](docs/spec/providers.md).

## Quickstart

idia is not on PyPI yet, so install it from GitHub with the extra for the provider you
call; it moves to PyPI later, and the extras keep their names:

```sh
pip install "idia[system-one] @ git+https://github.com/CRSiegfried/idia"
pip install "idia[openai] @ git+https://github.com/CRSiegfried/idia"
pip install "idia[cloudflare] @ git+https://github.com/CRSiegfried/idia"
```

System One is TypeSafe's decision-model API, whose models are named `jev-*`. Each provider
reads its credentials from the environment unless you pass them: System One reads
`TYPESAFE_API_KEY`, OpenAI reads `OPENAI_API_KEY`, and Cloudflare reads
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. idia reads no `.env` file, so
export them first.

A yes/no question and a choice, put to Cloudflare's Clef in one request:

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

An extraction, put to Cloudflare's Clairic:

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

`cloudflare()`, from the `idia.cloudflare` module, is one provider for both: a router
that sends the label kinds to Clef and extractions to Clairic, each its share of the request:

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

Each answer is a `NoulAnswer`, `LabelAnswer`, `ExtractionAnswer`, `BoxAnswer`, `Refusal` or
`Fault`; a fault is scoped to its question, and its siblings stand. A request that does not
fit the provider raises `DoesNotFit` before any call. When no wire request answers, `answer`
raises `RequestFailed`, which carries the response and its outcomes:

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

## Deciding with ommat

ommat binds an idia provider as an oracle through its own `ommat[idia]` extra,
taking one leaf's dispatcher per oracle. See ommat's tutorial,
[Decide with idia](https://github.com/CRSiegfried/ommat/blob/main/docs/tutorials/decide-with-idia.md).

## Development

```sh
uv sync --all-extras
```

Then run the checks in [CONTRIBUTING.md](CONTRIBUTING.md#development), which also gives the
ground rules; [docs/spec](docs/spec/README.md) is the normative description.
