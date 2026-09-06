@echo off
setlocal
cd /d "%~dp0"
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"
if not exist ".venv\floptician-backend.txt" (
    echo ERROR: Run setup.bat first. See SETUP.md for migration instructions.
    if not defined FLOPTICIAN_NO_PAUSE pause
    exit /b 1
)
REM Check metadata without resolving versions or changing installed packages.
uv lock --check --offline
if errorlevel 1 (
    echo ERROR: The project lock needs attention. Run setup.bat or see SETUP.md.
    if not defined FLOPTICIAN_NO_PAUSE pause
    exit /b 1
)
uv run --no-sync floptician run %*
set "run_result=%errorlevel%"
if not defined FLOPTICIAN_NO_PAUSE pause
exit /b %run_result%
