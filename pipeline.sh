#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
artifact_directory=$(mktemp -d)
trap 'rm -rf "$artifact_directory"' EXIT
cd "$repository_root"

poetry install --with dev --extras plot --no-interaction
./scripts/check.sh
poetry run pytest \
  --cov=mc_dagprop \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=85

poetry run python -m build --wheel --sdist --outdir "$artifact_directory"
for artifact in "$artifact_directory"/*.whl "$artifact_directory"/*.tar.gz; do
  poetry run python scripts/test_installed_artifact.py "$artifact"
done

poetry run python scripts/run_readme_examples.py
poetry run python -m demo.analytic
poetry run python -m demo.monte_carlo
poetry run python -m demo.distribution --trials 100 --no-show
