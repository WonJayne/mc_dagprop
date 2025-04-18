@echo off
setlocal
REM Exit immediately if any command fails

REM Delete old build artifacts
del /q dist\*.*
del /q build\*.*

call python setup.py build_ext --inplace || exit /b
call python -m build --wheel || exit /b

for %%f in (dist\mc_dagprop*.whl) do (
    call pip install "%%f" --force-reinstall || exit /b
)

python test\test_simulator.py || exit /b