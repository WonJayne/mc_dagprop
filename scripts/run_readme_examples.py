"""Execute every Python code block in README.md in an isolated subprocess."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_python_blocks(readme: str) -> tuple[str, ...]:
    return tuple(block.strip() for block in PYTHON_BLOCK.findall(readme) if block.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("readme", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "README.md")
    args = parser.parse_args()

    blocks = extract_python_blocks(args.readme.read_text(encoding="utf-8"))
    if not blocks:
        raise RuntimeError(f"no executable Python blocks found in {args.readme}")

    with tempfile.TemporaryDirectory(prefix="mc-dagprop-readme-") as temporary_directory:
        for block_number, block in enumerate(blocks, start=1):
            try:
                subprocess.run([sys.executable, "-I", "-c", block], cwd=temporary_directory, check=True, timeout=60)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"README Python block {block_number} failed") from exc
    print(f"executed {len(blocks)} README Python blocks")


if __name__ == "__main__":
    main()
