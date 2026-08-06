"""Require a release tag to match the PEP 621 package version exactly."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(?:a|b|rc)(0|[1-9]\d*))?$")


def read_project_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as pyproject_file:
        version = str(tomllib.load(pyproject_file)["project"]["version"])
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"unsupported release version format in pyproject.toml: {version!r}")
    return version


def require_matching_tag(tag: str, version: str) -> None:
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(f"release tag {tag!r} does not match package version; expected {expected_tag!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--pyproject", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml")
    args = parser.parse_args()

    version = read_project_version(args.pyproject)
    require_matching_tag(args.tag, version)
    print(f"release tag {args.tag} matches package version {version}")


if __name__ == "__main__":
    main()
