@echo off
setlocal
REM Exit immediately if any command fails

REM Delete old build artifacts
del /q dist\*.*
del /q build\*.*

call poetry build || exit /b

for %%f in (dist\mc_dagprop*.whl) do (
    call pip install "%%f" --force-reinstall || exit /b
)

pip install plotly || exit /b

python mc_dagprop\utils\demonstration.py || exit /b

python test\test_simulator.py || exit /b