"""Check that the library, its docs and its commit subjects describe the library.

Run it as ``uv run python scripts/check_terms.py``. It scans ``src/``, ``docs/``,
``tests/`` (its recorded cassettes aside), ``scripts/`` (this script aside), ``skills/``,
``.github/``, the top-level README, CHANGELOG, CONTRIBUTING and SECURITY
files and ``pyproject.toml`` for the patterns in ``TERMS``, words that describe how a change
was organized rather than what the library does, and exits 1 with one line per hit, or 0
when none is found. Standard library only.

Each pattern is a whole word or a phrase, matched with word boundaries, so the ordinary
words beside it pass: ``registered`` and ``registry`` are not ``the register``, ``LeafHarness``
and "the test HTTP harness" are not matched, and "a ruled line" passes because only
``ruled on`` and ``ruled by`` are matched. A numbered round matches only when a lap, point,
item or step follows it, so a fixpoint's "round 3 of the loop" and a ``round_bound`` pass
while "round 5 lap 2" does not. A line that must use a term in its ordinary sense carries
``check-terms: ok`` and is skipped.

``TERMS`` is the module-level tuple of every pattern with its name, and ``hits`` applies them
to one line. ``--subjects <range>`` applies the same patterns to the subject of each commit
in a git revision range, such as ``main..HEAD``, instead of scanning files; CI runs it over
the commits a push to main brings.

Before scanning, the script checks itself: every planted sentence in ``PLANTED`` must be
caught and every sentence in ``ORDINARY`` must pass, so a pattern that stops matching, or
starts matching ordinary prose, fails the run.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TREES = ("src", "docs", "tests", "scripts", "skills", ".github")
SKIPPED = ("tests/cassettes/", "scripts/check_terms.py")
TOP_LEVEL = ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml")
SUFFIXES = frozenset({".py", ".md", ".pyi", ".txt", ".toml", ".json", ".yml", ".yaml"})
ALLOW_MARK = "check-terms: ok"

TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("office", re.compile(r"\boffices?\b", re.IGNORECASE)),
    ("charter", re.compile(r"\bcharters?\b", re.IGNORECASE)),
    ("the register", re.compile(r"\bthe register\b", re.IGNORECASE)),
    ("seam", re.compile(r"\bseams?\b", re.IGNORECASE)),
    ("ADR", re.compile(r"\bADRs?\b")),
    ("section sign", re.compile(r"§")),
    ("expert", re.compile(r"\bexperts?\b", re.IGNORECASE)),
    ("ruling", re.compile(r"\brulings?\b|\bruled (?:on|by)\b", re.IGNORECASE)),
    ("delegation", re.compile(r"\bdelegation\b|\buser'?s delegate\b", re.IGNORECASE)),
    ("finding for", re.compile(r"\bfindings? for the (?:user|office)\b", re.IGNORECASE)),
    ("WORKFLOW", re.compile(r"\bWORKFLOW(?:\.md)?\b")),
    ("friction", re.compile(r"\bfriction\b", re.IGNORECASE)),
    (
        "build round",
        re.compile(r"\bround ?\d+\b[\s,:;-]*(?:laps?|points?|items?|steps?)\b", re.IGNORECASE),
    ),
    ("lap", re.compile(r"\blaps?\b", re.IGNORECASE)),
    ("build step", re.compile(r"\bbuild steps?\b|\bstep ?\d+\b", re.IGNORECASE)),
    ("build item", re.compile(r"\bitems? ?\d+\b", re.IGNORECASE)),
    ("orchestration", re.compile(r"\borchestrat\w*", re.IGNORECASE)),
    ("decision record", re.compile(r"\bdecision records?\b", re.IGNORECASE)),
    ("build harness", re.compile(r"\bbuild harness\b", re.IGNORECASE)),
)

PLANTED: tuple[str, ...] = (
    "A harness is what a provider's office supplies.",
    "The charter says so.",
    "The seam is in the register.",
    "See the ADR on retries.",
    "As §some-slug says.",
    "Ask the expert first.",
    "Ruled on 1 May.",
    "Made on the owner's delegation.",
    "That is a finding for the user.",
    "Follow WORKFLOW.md.",
    "Friction point 3.",
    "Closed in round 2 point 3.",
    "Round 5 lap 2 point 5",
    "The lap closed.",
    "Wire leaves register here as their build steps land.",
    "Step 2 is a roll call.",
    "Item 4 closed.",
    "The items 3 and 4 are open.",
    "The orchestrator merges it.",
    "Orchestration of the build.",
    "The decision record holds it.",
    "It lives in the build harness.",
)

ORDINARY: tuple[str, ...] = (
    "Each wire leaf registers here; a registered name is in the registry.",
    "The leaf is registered as cloudflare-clef.",
    "A LeafHarness builds the leaf; the test HTTP harness writes the cassette.",
    "The checker ruled out the bad value on a ruled line.",
    "A leaf that delegates everything to inner.",
    "The workflow in ci.yml runs the checks.",
    "Each step of the call loop waits for the next.",
    "round(no, decimals) == no",
    "Round 0 is a rule pass; round 3 of the fixpoint asks the held sources.",
    "The round_bound is 8, and a source is asked again in round 2.",
    "An item of the tuple is checked; the items are typed.",
    "JSON holding a type key.",
    "A compound eye is made of many small eyes.",
)


SUBJECT_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("build round", re.compile(r"^(?:Merge )?round ?\d+", re.IGNORECASE)),
    ("build point", re.compile(r"\bpoints? ?\d+\b", re.IGNORECASE)),
    ("open-question label", re.compile(r"\bU\d+\b")),
    ("review label", re.compile(r"\b(?i:rule) [A-Z]\b|\bC\d+\b|\b(?i:addition) [A-Z]\b")),
)
"""Shapes a commit subject may not hold, besides ``TERMS``: in a subject a numbered round,
a numbered point or a short review label can only name a build step, while a file may use
such words in their ordinary sense."""

PLANTED_SUBJECTS: tuple[str, ...] = (
    "Round 6 event loop: L11 also over a second event loop",
    "Merge round6/loop: a leaf's client follows the running event loop",
    "Spec: one sentence per line (point 111)",
    "The U7 finding is recorded",
    "Rule A tests retry again",
    "C1: the chat leaf serves one question a call",
    "Addition A: the vocabulary says how wire ids read",
)

ORDINARY_SUBJECTS: tuple[str, ...] = (
    "Merge event-loop: a leaf's client follows the running event loop",
    "A round trip through the recorded leaf keeps the usage",
    "Conformance P11 accepts an outcome left unsent by a credential error",
    "Deadline.after refuses minus infinity with its own message",
    "The UTF-8 body of a request is counted in bytes",
)


def subject_hits(subject: str) -> list[str]:
    """The names of the terms and the subject-only shapes ``subject`` holds."""
    return hits(subject) + [name for name, pattern in SUBJECT_TERMS if pattern.search(subject)]


def subject_self_check() -> list[str]:
    """The file self-check, then one problem per planted subject missed and per ordinary
    subject caught."""
    problems = self_check()
    problems += [
        f"self-check: missed subject {s!r}" for s in PLANTED_SUBJECTS if not subject_hits(s)
    ]
    problems += [
        f"self-check: caught subject {s!r} as {subject_hits(s)}"
        for s in ORDINARY_SUBJECTS
        if subject_hits(s)
    ]
    return problems


def hits(line: str) -> list[str]:
    """The names of the terms ``line`` holds, none when it carries the allow mark."""
    if ALLOW_MARK in line:
        return []
    return [name for name, pattern in TERMS if pattern.search(line)]


def self_check() -> list[str]:
    """One problem per planted sentence the patterns miss and per ordinary one they catch."""
    problems = [f"self-check: missed {s!r}" for s in PLANTED if not hits(s)]
    problems += [f"self-check: caught {s!r} as {hits(s)}" for s in ORDINARY if hits(s)]
    return problems


def files(root: Path) -> list[Path]:
    """The files scanned under ``root``."""
    found = [root / name for name in TOP_LEVEL if (root / name).is_file()]
    for tree in TREES:
        found += sorted(
            p
            for p in (root / tree).rglob("*")
            if p.is_file()
            and p.suffix in SUFFIXES
            and not p.relative_to(root).as_posix().startswith(SKIPPED)
        )
    return found


def scan(root: Path) -> list[str]:
    """One problem per line holding a term."""
    problems: list[str] = []
    for path in files(root):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for name in hits(line):
                problems.append(f"{path.relative_to(root).as_posix()}:{number}: {name}")
    return problems


def subjects(revisions: str, root: Path = ROOT) -> list[tuple[str, str]]:
    """Each commit's short hash and subject in ``revisions``, oldest first, by ``git log``."""
    listed = subprocess.run(
        ["git", "log", "--reverse", "--format=%h %s", "--end-of-options", revisions, "--"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        (commit, subject)
        for commit, _, subject in (line.partition(" ") for line in listed.stdout.splitlines())
    ]


def check_subjects(revisions: str, root: Path = ROOT) -> int:
    """One problem per commit subject in ``revisions`` holding a term or a subject-only
    shape; 1 when any does."""
    problems = subject_self_check()
    if not problems:
        try:
            found = subjects(revisions, root)
        except subprocess.CalledProcessError as error:
            print(
                f"check_terms: git log {revisions} failed: {error.stderr.strip()}", file=sys.stderr
            )
            return 2
        problems = [
            f"{commit}: {name}: {subject}"
            for commit, subject in found
            for name in subject_hits(subject)
        ]
        if not problems:
            print(f"check_terms: {len(found)} commit subjects in {revisions} pass")
            return 0
    for problem in problems:
        print(problem)
    print(f"check_terms: {len(problems)} problem(s)", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if argv[:1] == ["--subjects"]:
        if len(argv) != 2:
            print("usage: check_terms.py --subjects <revision range>", file=sys.stderr)
            return 2
        return check_subjects(argv[1])
    root = Path(argv[0]) if argv else ROOT
    problems = self_check()
    if not problems:
        problems = scan(root)
    for problem in problems:
        print(problem)
    if problems:
        print(f"check_terms: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"check_terms: {len(files(root))} files pass; the self-check caught {len(PLANTED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
