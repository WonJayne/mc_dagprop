#!/usr/bin/env bash
set -euo pipefail

# Delete old build artifacts
rm -f dist/*
rm -f build/*

poetry build

for f in dist/mc_dagprop*.whl; do
    pip install "$f" --force-reinstall
done

pip install plotly

python test/test_simulator.py

