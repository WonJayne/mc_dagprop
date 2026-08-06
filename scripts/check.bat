@echo off
setlocal

echo Running Ruff on configured scope...
call poetry run ruff check . || exit /b 1

echo Running Black (check mode) on configured scope...
call poetry run black --check . || exit /b 1

echo Running BasedPyright on the package and public-consumer scopes...
call poetry run basedpyright || exit /b 1

echo Checks completed.
