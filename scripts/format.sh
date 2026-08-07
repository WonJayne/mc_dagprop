#!/usr/bin/env bash
set -euo pipefail

echo "Running Black (fix mode) on configured scope..."
poetry run black .

echo "Running Ruff (fix mode) on configured scope..."
poetry run ruff check --fix .

echo "Formatting completed."
