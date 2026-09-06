@echo off
setlocal
cd /d "%~dp0"
REM Keep this launcher's environment local even when called from another project.
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"

uv --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Install uv first. See SETUP.md.
    goto :failed
)

REM Never let uv replace an existing, unmanaged environment during migration.
if exist ".venv" if not exist ".venv\floptician-backend.txt" (
    echo ERROR: An existing .venv needs migration. See SETUP.md.
    echo Rename it to an unused backup name before running setup again.
    goto :failed
)

set "backend=%~1"
if not defined backend if exist ".venv\floptician-backend.txt" set /p backend=<".venv\floptician-backend.txt"
if not defined backend (
    set "backend=cpu"
    nvidia-smi >nul 2>&1
    if not errorlevel 1 set "backend=cuda"
)
if not "%backend%"=="cpu" if not "%backend%"=="cuda" (
    echo ERROR: Use setup.bat, setup.bat cpu, or setup.bat cuda.
    goto :failed
)

echo Installing the locked %backend% environment...
uv sync --locked --no-dev --extra windows --extra %backend%
if errorlevel 1 goto :failed

REM Record the installed build so an update preserves the CPU/CUDA choice.
>".venv\floptician-backend.txt" echo %backend%
if not exist "config.yaml" copy config.example.yaml config.yaml >nul
if errorlevel 1 goto :failed

uv run --no-sync floptician validate-config
if errorlevel 1 (
    echo Fix the configuration or download the model with git lfs pull, then retry.
    goto :failed
)
uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 goto :failed
if "%backend%"=="cuda" (
    uv run --no-sync python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if errorlevel 1 (
        echo ERROR: CUDA is unavailable. Update the NVIDIA driver or use setup.bat cpu.
        goto :failed
    )
)
echo Setup complete. See SETUP.md for OBS setup, then run run.bat.
if not defined FLOPTICIAN_NO_PAUSE pause
exit /b 0

:failed
echo Setup did not complete. Review the error above.
if not defined FLOPTICIAN_NO_PAUSE pause
exit /b 1
