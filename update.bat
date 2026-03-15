@echo off
cd /d "%~dp0"
echo.
echo ========================================
echo   Floptician Update
echo ========================================
echo.

REM --- Pull latest code ---
echo Pulling latest changes...
git pull
if errorlevel 1 (
    echo.
    echo ERROR: Failed to pull updates.
    echo Make sure you have internet access and Git is installed.
    pause
    exit /b 1
)

REM --- Activate venv and reinstall ---
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo Updating dependencies...
pip install -e ".[windows]"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to update dependencies.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Update complete!
echo ========================================
echo.
pause
