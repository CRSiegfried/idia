"""The README's keyless demo runs and prints what the README shows, and the tutorial's
examples compile.

The demo is the Python code between the README's demo markers, run in a fresh interpreter;
the output it shows is the text block after that code. The tutorial's examples make live
calls, so they are compiled, never run."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
TUTORIAL = ROOT / "docs" / "tutorials" / "ask-a-provider.md"
DEMO = re.compile(r"<!-- demo -->\n(.*?)<!-- /demo -->", re.DOTALL)
BLOCK = re.compile(r"```(python|text)\n(.*?)```", re.DOTALL)


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
    blocks = BLOCK.findall(TUTORIAL.read_text(encoding="utf-8"))
    code = [body for kind, body in blocks if kind == "python"]
    assert code, "the tutorial holds Python blocks"
    for number, body in enumerate(code, start=1):
        compile(body, f"{TUTORIAL.name} block {number}", "exec")
