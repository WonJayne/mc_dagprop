#!/usr/bin/env bash
set -euo pipefail

# Delete old build artifacts
rm -rf dist/*
rm -rf build/*


poetry build

BUILD_LIB_DIR=$(find build -type d -name 'lib.*' | head -n 1)
cp "$BUILD_LIB_DIR/mc_dagprop/monte_carlo/"*_core*.so mc_dagprop/monte_carlo/

pip install dist/mc_dagprop*.whl --force-reinstall

cd test
python -m unittest discover -s . -p "test_*.py"

