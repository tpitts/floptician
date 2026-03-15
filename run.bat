@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
floptician run
pause
