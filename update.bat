@echo off
setlocal
cd /d "%~dp0"
echo Pulling the latest code...
git pull --ff-only
if errorlevel 1 (
    echo ERROR: Update stopped. Check the connection and local Git changes.
    echo Diverged branches must be resolved manually; no automatic merge was made.
    if not defined FLOPTICIAN_NO_PAUSE pause
    exit /b 1
)
REM Reuse setup so updates install the same locked dependencies and backend.
call setup.bat
exit /b %errorlevel%
