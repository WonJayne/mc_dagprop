@echo off
setlocal
REM Exit immediately if any command fails

REM Delete old build artifacts
del /q dist\*.*
del /q build\*.*


call poetry build || exit /b

for %%f in (dist\*.whl) do (
    call pip install %%f --force-reinstall || exit /b
)

call python demo/distribution.py || exit /b

python -m unittest discover -s test || exit /b
