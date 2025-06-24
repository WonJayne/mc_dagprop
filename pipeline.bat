@echo off
setlocal
REM Exit immediately if any command fails

REM Delete old build artifacts
del /q dist\*.*
del /q build\*.*


call poetry build || exit /b

for /d %%d in (build\lib.*) do (
    copy "%%d\mc_dagprop\monte_carlo\*_core*.so" mc_dagprop\monte_carlo\ >nul
)

for %%f in (dist\*.whl) do (
    call pip install %%f --force-reinstall || exit /b
)

call python mc_dagprop\utils\demonstration.py || exit /b

cd test
python -m unittest discover -s . -p "test_*.py" || exit /b
