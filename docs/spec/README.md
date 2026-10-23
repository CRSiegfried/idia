# The idia spec

The normative description of what idia does: the shapes a request and a response take, what a provider declares and how a request is checked against it, how the questions are packed when they overflow, how one deadline bounds every wait and retry, what each request outcome records, and the rules every answer passes.
The code follows this document; where they differ, one of them has a bug, and the tests say which.

## Sections

- [Vocabulary](vocabulary.md): state, question, kind, answer, request, response, validation with its rounding tolerances, and the two protocols.
- [Providers and capabilities](providers.md): the HTTP base and its status table, and each wire leaf's body, reading and record: System One, OpenAI's decisions leaf and its chat leaf, and Cloudflare's Clef and Clairic with the router the `cloudflare` extra returns.
- [Fit](fit.md): preflight, the limits and their warnings, packing, state overflow and the overflow policy, and the merge.
- [Dispatch](dispatch.md): the deadline scope, the retry policy and the limits, the outcome and its attempts, the three endings, the re-ask, and the router and fallback with their derived record.
- [Elicitation from a chat model](elicitation.md): the `logprobs` read protocol of the chat leaf, its faults, its derived figures and what it records.
- [Test doubles](test-doubles.md): the scripted leaf, the recorded leaf with its modes and log, and the HTTP harness and its cassettes in idia's tests.
- [The conformance suite](conformance.md): the leaf half and the provider half, the harness and its exchanges, the three verdicts, and how a leaf's tests run it.
