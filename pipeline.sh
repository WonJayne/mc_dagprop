#!/usr/bin/env bash
set -euo pipefail

# Delete old build artifacts
rm -rf dist/*
rm -rf build/*

poetry build

pip install dist/mc_dagprop*.whl --force-reinstall

python -m unittest discover -s test

