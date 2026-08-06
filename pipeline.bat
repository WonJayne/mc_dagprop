@echo off
setlocal

cd /d "%~dp0"
set "artifact_directory=%TEMP%\mc-dagprop-artifacts-%RANDOM%-%RANDOM%"
mkdir "%artifact_directory%" || exit /b 1

call poetry install --with dev --extras plot --no-interaction || exit /b 1
call scripts\check.bat || exit /b 1
call poetry run pytest --cov=mc_dagprop --cov-branch --cov-report=term-missing --cov-fail-under=85 || exit /b 1
call poetry run python -m build --wheel --sdist --outdir "%artifact_directory%" || exit /b 1

for %%f in ("%artifact_directory%\*.whl") do (
    call poetry run python scripts\test_installed_artifact.py "%%f" || exit /b 1
)
for %%f in ("%artifact_directory%\*.tar.gz") do (
    call poetry run python scripts\test_installed_artifact.py "%%f" || exit /b 1
)

call poetry run python scripts\run_readme_examples.py || exit /b 1
call poetry run python -m demo.analytic || exit /b 1
call poetry run python -m demo.monte_carlo || exit /b 1
call poetry run python -m demo.distribution --trials 100 --no-show || exit /b 1

rmdir /s /q "%artifact_directory%"
