@echo off
setlocal

set "option=%~1"
if "%option%"=="-h" goto usage
if "%option%"=="--help" goto usage
if "%option%"=="-s" set "option=--check"
if "%option%"=="--score" set "option=--check"
if "%option%"=="-c" set "option=--check"
if "%option%"=="-r" set "option=--reformat"
if "%option%"=="-t" set "option=--test"
if "%option%"=="-a" set "option=--all"

if "%option%"=="--check" (
    call scripts\check.bat || exit /b 1
    exit /b 0
)

if "%option%"=="--reformat" (
    call scripts\format.bat || exit /b 1
    exit /b 0
)

if "%option%"=="--test" (
    call poetry run pytest || exit /b 1
    exit /b 0
)

if "%option%"=="--all" (
    call scripts\format.bat || exit /b 1
    call scripts\check.bat || exit /b 1
    call poetry run pytest || exit /b 1
    exit /b 0
)

:usage
echo.
echo OpenBus-compatible development checks
echo.
echo Usage: devtools.cmd [OPTION]
echo.
echo Options:
echo --check, -c       Run Ruff, Black check, and BasedPyright
echo --reformat, -r    Apply Ruff fixes and Black formatting
echo --test, -t        Run pytest
echo --all, -a         Reformat, check, and test
echo --score, -s       Compatibility alias for --check
echo -h, --help        Display this help message
exit /b 0
