"""Check the spec in docs/spec: links, headings and the index.

Run it as ``uv run python scripts/check_spec.py [directory]``, the directory docs/spec by
default. It exits 0 when the spec passes and 1 with one line per problem otherwise. Standard
library only.

- Links: every relative link resolves to a file, and every ``#anchor`` to a heading in the
  file it names (GitHub's rule: lowercase, punctuation dropped, spaces to hyphens).
- Headings: one level-1 heading, first; no level skipped going down; no heading twice.
- Index: README.md's Sections list links every other spec file, and each entry links one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "spec"
INDEX = "README.md"
INDEX_SECTION = "Sections"

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
LINK = re.compile(r"(?<!!)\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^()\s]+)(?:\s+\"[^\"]*\")?\)")
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def prose_lines(text: str) -> list[tuple[int, str]]:
    """The numbered lines outside fenced code blocks."""
    lines: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            lines.append((number, line))
    return lines


def headings(text: str) -> list[tuple[int, int, str]]:
    """Each heading as (line number, level, text)."""
    found: list[tuple[int, int, str]] = []
    for number, line in prose_lines(text):
        match = HEADING.match(line)
        if match:
            found.append((number, len(match.group(1)), match.group(2)))
    return found


def anchor(heading: str) -> str:
    """GitHub's anchor for a heading's text."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(text: str) -> set[str]:
    """Every anchor a file's headings give, with GitHub's suffixes for repeats."""
    seen: dict[str, int] = {}
    result: set[str] = set()
    for _, _, heading in headings(text):
        base = anchor(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def links(text: str) -> list[tuple[int, str]]:
    """Each link target outside code, as (line number, target)."""
    found: list[tuple[int, str]] = []
    for number, line in prose_lines(text):
        bare = CODE_SPAN.sub("", line)
        found.extend((number, match.group(1)) for match in LINK.finditer(bare))
    return found


def check_links(path: Path, text: str) -> list[str]:
    problems: list[str] = []
    for number, target in links(text):
        if EXTERNAL.match(target):
            continue
        file_part, _, fragment = target.partition("#")
        destination = (path.parent / file_part).resolve() if file_part else path
        where = f"{path.name}:{number}"
        if not destination.is_file():
            problems.append(f"{where}: link to {target!r}: no such file")
            continue
        if (
            fragment
            and destination.suffix == ".md"
            and fragment not in anchors(destination.read_text(encoding="utf-8"))
        ):
            problems.append(f"{where}: link to {target!r}: no such heading")
    return problems


def check_headings(path: Path, text: str) -> list[str]:
    problems: list[str] = []
    found = headings(text)
    if not found or found[0][1] != 1:
        problems.append(f"{path.name}: the first heading is not level 1")
    level_ones = [number for number, level, _ in found if level == 1]
    if len(level_ones) > 1:
        problems.append(f"{path.name}:{level_ones[1]}: a second level-1 heading")
    previous = 0
    seen: dict[str, int] = {}
    for number, level, heading in found:
        if previous and level > previous + 1:
            problems.append(f"{path.name}:{number}: level {level} under level {previous}")
        previous = level
        key = heading.strip().lower()
        if key in seen:
            problems.append(f"{path.name}:{number}: {heading!r} repeats line {seen[key]}")
        else:
            seen[key] = number
    return problems


def check_index(spec: Path) -> list[str]:
    index = spec / INDEX
    if not index.is_file():
        return [f"{INDEX}: missing"]
    problems: list[str] = []
    in_section = False
    entries: list[tuple[int, str]] = []
    for number, line in prose_lines(index.read_text(encoding="utf-8")):
        match = HEADING.match(line)
        if match:
            in_section = match.group(2) == INDEX_SECTION
            continue
        if in_section and re.match(r"^\s*[-*]\s", line):
            entries.append((number, line))
    if not entries:
        return [f"{INDEX}: no entries under '## {INDEX_SECTION}'"]
    named: dict[str, int] = {}
    for number, line in entries:
        targets = [t for _, t in links(line) if not EXTERNAL.match(t)]
        files = {t.partition("#")[0] for t in targets if t.partition("#")[0]}
        if len(files) != 1:
            problems.append(f"{INDEX}:{number}: an entry links {len(files)} spec files, not 1")
            continue
        name = files.pop()
        if name in named:
            problems.append(f"{INDEX}:{number}: {name} is listed again (line {named[name]})")
        named[name] = number
    expected = {p.name for p in spec.glob("*.md") if p.name != INDEX}
    for name in sorted(expected - named.keys()):
        problems.append(f"{INDEX}: {name} is not in the index")
    for name in sorted(named.keys() - expected):
        problems.append(f"{INDEX}:{named[name]}: {name} is not a spec file")
    return problems


def shown(spec: Path) -> str:
    """The spec directory as the summary names it: relative to the repo root when under it."""
    resolved = spec.resolve()
    if resolved.is_relative_to(ROOT):
        return resolved.relative_to(ROOT).as_posix()
    return str(spec)


def main(argv: list[str]) -> int:
    spec = Path(argv[0]) if argv else SPEC
    problems: list[str] = []
    files = sorted(spec.glob("*.md"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        problems += check_links(path, text)
        problems += check_headings(path, text)
    problems += check_index(spec)
    for problem in problems:
        print(problem)
    if problems:
        print(f"check_spec: {len(problems)} problem(s) in {shown(spec)}", file=sys.stderr)
        return 1
    print(f"check_spec: {len(files)} spec files in {shown(spec)} pass (links, headings, index)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
