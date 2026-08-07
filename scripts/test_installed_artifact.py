"""Install and smoke-test one wheel or source distribution in a clean environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _venv_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    artifact = args.artifact.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    smoke_script = source_root / "scripts" / "smoke_installed_distribution.py"

    with tempfile.TemporaryDirectory(prefix="mc-dagprop-artifact-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        environment = temporary_path / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = _venv_python(environment)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", str(artifact)],
            cwd=temporary_path,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", str(smoke_script), "--source-root", str(source_root)], cwd=temporary_path, check=True
        )


if __name__ == "__main__":
    main()
