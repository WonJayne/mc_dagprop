@echo off
setlocal

echo Running Black (fix mode) on configured scope...
call poetry run black . || exit /b 1

echo Running Ruff (fix mode) on configured scope...
call poetry run ruff check --fix . || exit /b 1

echo Formatting completed.
