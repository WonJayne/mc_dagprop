#!/usr/bin/env bash
set -euo pipefail

rm -rf dist/* build/*
python -m pytest -q
python -m build --wheel
wheel_path=$(python - <<'PY'
from pathlib import Path
print(next(Path('dist').glob('mc_dagprop*.whl')).resolve())
PY
)
smoke_dir=$(mktemp -d)
python -m venv "$smoke_dir/venv"
"$smoke_dir/venv/bin/python" -m pip install --upgrade pip >/dev/null
"$smoke_dir/venv/bin/python" -m pip install "$wheel_path" >/dev/null
(
  cd "$smoke_dir"
  "$smoke_dir/venv/bin/python" /workspace/mc_dagprop/scripts/smoke_installed_wheel.py
)
python -m demo.analytic
