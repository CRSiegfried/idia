# Elicitation from a chat model

An elicitation is the named protocol by which a leaf turns a model's output into answers.
Every wire response names one.
A decision model reports `passthrough`: its probabilities are the wire's own.
This release admits one other, `logprobs`, which a chat leaf reports.
`structured_output`, `verbalised` and `sampled_n` are candidates: a chat leaf built with one raises `ElicitationNotInRelease(elicitation, candidate=True)`, and any other unknown name raises it with `candidate=False`.
Changing the `logprobs` protocol changes recorded answers, so it is a breaking change; reading an absent letter as zero mass rather than as a fault is part of the protocol.

## The `logprobs` protocol

One question per call, since one forward pass yields one distribution.
A request of k questions is k wire requests, each repeating the state.

1. **Letters.** The labels asked are coded as single letters in the order asked, A for the first.
   A `noul` is coded A for yes and B for no.
   At most 20 letters, the top-k the wire allows.
2. **The prompt** asks for the letter alone: a system message, then the state, the instruction and one line per letter, `A. <name>` or `A. <name>: <description>`.
   The temperature is 0, `max_completion_tokens` is 1, and no `logit_bias` and no reasoning parameter is sent.
   Thinking is off because the leaf reads only models that answer without it: OpenAI's reasoning models refuse `logprobs`.
3. **The read.** Only the first answer token's top logprobs are read, with the top-k the wire allows requested.
   A letter's mass is the sum of `exp(logprob)` over its surface forms present: the letter itself, with a leading space, and either case of each (`A`, ` A`, `a`, ` a`).
   A token the wire lists twice is summed.
   Every sum in the read, inside the log-space sums, for the off-label mass and for the expected level, is taken with `math.fsum`, which rounds once whatever the order, so the read gives the same floats on every supported Python version.
4. **Renormalization** over the labels asked is part of the protocol and happens before validation: each label's probability is its mass over the sum of the label masses, computed in log space (`exp(lp_i - logsumexp(lp))` over the letters present), so letters far down the top-k never sum to zero.
   A logprob that is not a finite number at most zero is a `kind_shape` fault on that question.
   The mass outside the labels, one less the sum of the label masses and never below zero, is recorded as the answer's `off_label_mass`.
   It is a signal of the model answering something else, not a probability of any answer.
5. **The answer.**
   - `noul`: `NoulAnswer(probability)`, A's share of A and B.
     No second value is kept and `derived` is empty.
   - `choice`: `LabelAnswer` with the renormalized probabilities in label order and `chosen`, the first label of highest probability; `derived` is `{"chosen"}`.
   - `score`: as a `choice`, plus `expected_level`, the sum of each level's index times its probability; `derived` is `{"chosen", "expected_level"}`.
   - `confidence` is never set: a chat leaf has no provider confidence.
6. **Validation** runs over every answer, the derived figures included, with rounding none, so the sum tolerance is 1e-6 and a derived figure that disagreed with the probabilities would be a fault.

## Faults

A letter in none of its surface forms among the top-k reads as zero mass: its mass is at most the smallest probability returned, and on a confident answer look-alike tokens can fill the top-k.
The answer stands, and the wire response carries a `label-absent-from-logprobs` warning whose `field` is the question's key and whose message names each absent label with its letter in brackets, and that ceiling.

Three conditions are a `Fault` on that question, never a guess.
The siblings of a fault stand, and nothing is retried.

| Condition | Rule | Observed |
|---|---|---|
| no logprobs: none returned, no content token, or an empty top-k | `empty_logprobs` | none |
| no label's letter, in any surface form, among the top-k | `label_absent_from_logprobs`, one failure per letter, gathered in one fault | the letter |
| the off-label mass above the bound the leaf was built with | `off_label_mass_above_bound`, the answer kept as mapped | the mass, against the bound |

The bound, `max_off_label_mass`, is unbounded by default.
A re-ask after a fault is the dispatcher's option, off by default.

## What the response records

The wire response names `logprobs` and declares `calibrated` false: the distribution is the model's own next-token distribution over the letters, not a calibrated probability, and a caller may apply its own temperature scaling by the elicitation's name.
Its `prompt_sent` records what fixes the result: the messages, the model, the temperature, the answer length, the top-k requested and the number returned, the letter map, and the bound.
A chat leaf's record declares `max_questions` 1, its label and level caps at the top-k, rounding none, no confidence and `calibrated` false.
