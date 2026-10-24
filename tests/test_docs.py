"""The shipped docs' code works and their links hold.

- The README's keyless demo, the Python between its demo markers, runs in a fresh
  interpreter and prints the text block that follows it.
- The tutorial's examples make live calls, so they are compiled, never run.
- The how-to guide on writing a leaf is a working example in several files: each Python
  block's first line names the file it belongs to, the blocks of one file joined in order.
  Its `main.py` prints the text block that follows it, and its tests pass under pytest.
- The guide on adding a provider quotes idia's own files: each Python, TOML or YAML block's first
  line names the file, and the rest of the block appears in that file as written.
- The agent skill in skills/idia has a frontmatter with a name and a description, its Python
  compiles, and every link is absolute, pinned to this version's tag, and names a file of
  this repository and, where it has one, a heading of that file. The skill is copied out of
  the repository, so a relative link would break. A repeated heading, which GitHub numbers,
  is not matched.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
TUTORIAL = ROOT / "docs" / "tutorials" / "ask-a-provider.md"
SKILL = ROOT / "skills" / "idia" / "SKILL.md"
WRITE_A_LEAF = ROOT / "docs" / "how-to" / "write-a-leaf.md"
ADD_A_PROVIDER = ROOT / "docs" / "how-to" / "add-a-provider-to-idia.md"
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
PINNED = f"https://github.com/CRSiegfried/idia/blob/v{VERSION}/"

DEMO = re.compile(r"<!-- demo -->\n(.*?)<!-- /demo -->", re.DOTALL)
BLOCK = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
HEADING = re.compile(r"^#{1,6} (.+)$", re.MULTILINE)
LINK = re.compile(r"\]\(([^)\s]+)\)")
FILE_LINE = re.compile(r"# ([\w-]+\.py)\n")
PATH_LINE = re.compile(r"# ([\w./-]+)\n")


def python_blocks(text: str) -> list[str]:
    return [body for kind, body in BLOCK.findall(text) if kind == "python"]


def compile_blocks(path: Path) -> None:
    blocks = python_blocks(path.read_text(encoding="utf-8"))
    assert blocks, f"{path.name} holds Python blocks"
    for number, body in enumerate(blocks, start=1):
        compile(body, f"{path.name} block {number}", "exec")


def demo_blocks() -> tuple[str, str]:
    """The demo's code, its Python blocks joined, and the output its text block shows."""
    sections = DEMO.findall(README.read_text(encoding="utf-8"))
    assert len(sections) == 1, "the README holds one demo between its markers"
    blocks = BLOCK.findall(sections[0])
    code = [body for kind, body in blocks if kind == "python"]
    shown = [body for kind, body in blocks if kind == "text"]
    assert code, "the demo holds Python blocks"
    assert len(shown) == 1, "the demo holds one text block, its output"
    assert blocks[-1][0] == "text", "the output follows the code"
    return "\n".join(code), shown[0]


def frontmatter(text: str) -> dict[str, str]:
    """The `key: value` lines between the opening and closing `---`."""
    assert text.startswith("---\n"), "the skill opens with its frontmatter"
    head, sep, _ = text[4:].partition("\n---\n")
    assert sep, "the frontmatter closes with ---"
    fields: dict[str, str] = {}
    for line in head.splitlines():
        key, sep, value = line.partition(": ")
        assert sep, f"a frontmatter line is `key: value`: {line!r}"
        assert ": " not in value, f"{key} holds ': ', which plain YAML misreads"
        assert not value.startswith(("'", '"', "#", "[", "{")), f"{key} starts with YAML syntax"
        fields[key] = value.strip()
    return fields


def anchor(heading: str) -> str:
    """The anchor GitHub gives a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_readme_demo_prints_what_the_readme_shows(tmp_path: Path) -> None:
    code, shown = demo_blocks()
    path = tmp_path / "demo.py"
    path.write_text(code, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", path],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == shown


def test_tutorial_examples_compile() -> None:
    compile_blocks(TUTORIAL)


def test_skill_frontmatter_names_it_and_says_when_to_use_it() -> None:
    fields = frontmatter(SKILL.read_text(encoding="utf-8"))
    assert fields.get("name") == "idia"
    assert fields.get("description")


def test_skill_examples_compile() -> None:
    compile_blocks(SKILL)


def test_skill_links_are_pinned_and_name_files_that_exist() -> None:
    links = LINK.findall(SKILL.read_text(encoding="utf-8"))
    assert links, "the skill links into the docs"
    for link in links:
        assert link.startswith(PINNED), f"{link} is not pinned to v{VERSION}"
        path, _, fragment = link.removeprefix(PINNED).partition("#")
        target = ROOT / path
        assert target.is_file(), f"{link} names no file"
        if fragment:
            text = FENCE.sub("", target.read_text(encoding="utf-8"))
            assert fragment in {anchor(h) for h in HEADING.findall(text)}, (
                f"{link} names no heading"
            )


def guide_files(path: Path, into: Path) -> dict[str, str]:
    """Write each Python block of ``path`` to the file its first line names, the blocks of
    one file joined in order; return each file's text."""
    files: dict[str, list[str]] = {}
    for number, body in enumerate(python_blocks(path.read_text(encoding="utf-8")), start=1):
        named = FILE_LINE.match(body)
        assert named, f"{path.name} block {number} does not name its file on its first line"
        files.setdefault(named.group(1), []).append(body[named.end() :])
    written = {name: "\n\n".join(bodies) for name, bodies in files.items()}
    for name, text in written.items():
        (into / name).write_text(text, encoding="utf-8")
    return written


def test_write_a_leaf_main_prints_what_the_guide_shows(tmp_path: Path) -> None:
    guide_files(WRITE_A_LEAF, tmp_path)
    text = WRITE_A_LEAF.read_text(encoding="utf-8")
    shown = re.compile(r"```text\n(.*?)```", re.DOTALL).search(text, text.index("# main.py"))
    assert shown, "a text block follows main.py"
    done = subprocess.run(
        [sys.executable, "-X", "utf8", "main.py"],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == shown.group(1)


def test_write_a_leaf_tests_pass(tmp_path: Path) -> None:
    files = guide_files(WRITE_A_LEAF, tmp_path)
    assert {"spanleaf.py", "test_spanleaf.py", "conftest.py", "test_real_model.py"} <= set(files)
    done = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "failed" not in done.stdout
    assert "SPANLEAF_REAL_MODEL" in done.stdout, "the real-model test is skipped by its gate"


def test_add_a_provider_quotes_idias_files_as_they_are() -> None:
    blocks = [
        body
        for kind, body in BLOCK.findall(ADD_A_PROVIDER.read_text(encoding="utf-8"))
        if kind in ("python", "toml", "yaml")
    ]
    assert blocks, "the guide quotes idia's files"
    for body in blocks:
        named = PATH_LINE.match(body)
        assert named, f"a quoted block does not name its file: {body[:60]!r}"
        source = (ROOT / named.group(1)).read_text(encoding="utf-8")
        assert body[named.end() :] in source, f"the quote of {named.group(1)} is not in it"
