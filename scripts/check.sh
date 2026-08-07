#!/usr/bin/env bash
set -euo pipefail

echo "Running Ruff on configured scope..."
poetry run ruff check .

echo "Running Black (check mode) on configured scope..."
poetry run black --check .

echo "Running BasedPyright on the package and public-consumer scopes..."
poetry run basedpyright

echo "Checks completed."
