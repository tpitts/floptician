@echo off
cd /d "%~dp0"
echo.
echo ========================================
echo   Floptician Setup
echo ========================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo.
    echo Install Python from https://www.python.org/downloads/
    echo IMPORTANT: Check "Add python.exe to PATH" during install.
    echo.
    pause
    exit /b 1
)
echo [OK] Python found
python --version

REM --- Check Git ---
git --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Git is not installed or not on PATH.
    echo You won't be able to pull updates without Git.
    echo Install Git from https://git-scm.com/download/win
    echo.
) else (
    echo [OK] Git found
)

REM --- Check Git LFS ---
git lfs version >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Git LFS is not installed.
    echo Model files (.pt) may not have downloaded correctly.
    echo Re-install Git and make sure Git LFS is checked.
    echo.
) else (
    echo [OK] Git LFS found
)

REM --- Create venv ---
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM --- Activate venv ---
call venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

REM --- Install PyTorch with CUDA ---
echo.
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected — installing CPU-only PyTorch.
    echo Detection will work but will be slower.
    echo.
    pip install torch torchvision
) else (
    echo NVIDIA GPU detected — installing PyTorch with CUDA support...
    echo This download is ~2.5 GB, it may take a while.
    echo.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install PyTorch.
    echo Check the error messages above.
    pause
    exit /b 1
)

REM --- Install dependencies ---
echo.
echo Installing Floptician and dependencies...
pip install -e ".[windows]"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Check the error messages above.
    pause
    exit /b 1
)

REM --- Copy config ---
if not exist "config.yaml" (
    echo.
    echo Creating config.yaml from template...
    copy config.example.yaml config.yaml >nul
    echo [OK] config.yaml created
) else (
    echo.
    echo [OK] config.yaml already exists (not overwritten)
)

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo.
echo Next steps:
echo   1. Set up OBS (see SETUP.md Steps 6-8)
echo   2. Double-click run.bat to start
echo.
pause
