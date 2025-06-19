#!/usr/bin/env bash
set -euo pipefail

# Delete old build artifacts
rm -f dist/*
rm -f build/*


poetry build

pip install dist/mc_dagprop*.whl --force-reinstall

python test/test_simulator.py

