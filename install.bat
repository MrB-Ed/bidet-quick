@echo off
REM ============================================================================
REM Bidet Quick — one-click setup
REM
REM What this does:
REM   1. Creates a Python virtual env in .venv next to this script
REM   2. Installs the pinned requirements into it
REM   3. Copies a Startup-folder shortcut so Bidet Quick launches at login
REM
REM Requires: Python 3.10 or newer on PATH. (Get it from python.org.)
REM ============================================================================

setlocal
cd /d "%~dp0"

echo.
echo === Bidet Quick installer ===
echo Repo dir: %CD%
echo.

REM ---- 1. Find a usable Python ------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/windows/
    echo and make sure "Add Python to PATH" is checked during install.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version') do set PYVER=%%v
echo Found Python %PYVER%

REM ---- 2. Create the venv -----------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo Virtual env already exists at .venv  --  skipping creation.
) else (
    echo Creating virtual env in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] python -m venv failed.
        pause
        exit /b 1
    )
)

REM ---- 3. Install requirements ------------------------------------------------
echo.
echo Upgrading pip in the venv ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    pause
    exit /b 1
)

echo.
echo Installing requirements (this can take 5-10 minutes on first run, the GPU
echo DLL packages alone are ~700 MB) ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install -r requirements.txt failed.
    echo See the error above. Common causes:
    echo   - No internet connection
    echo   - Out of disk space
    echo   - Antivirus blocking pip
    pause
    exit /b 1
)

REM ---- 4. Install Startup shortcut -------------------------------------------
echo.
echo Installing Startup shortcut so Bidet Quick auto-launches at login ...

set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
copy /Y "run_whisper_stt.vbs" "%STARTUP_DIR%\Bidet Quick.vbs" >nul
if errorlevel 1 (
    echo [WARN] Could not copy Startup shortcut. You can still launch manually
    echo        with run_whisper_stt.bat or run_whisper_stt.vbs.
) else (
    echo OK -- Startup shortcut installed at:
    echo   %STARTUP_DIR%\Bidet Quick.vbs
)

REM ---- 5. Done ----------------------------------------------------------------
echo.
echo ============================================================================
echo  Done. Bidet Quick is installed.
echo.
echo  First-run notes:
echo   - The first time you press the hotkey, Windows may ask for microphone
echo     permission. Click Allow.
echo   - The first transcription downloads the Whisper large-v3 model (~3 GB).
echo     This is a one-time download; everything is offline after that.
echo   - Look for the gray dot in your system tray (overflow / up-arrow).
echo     Gray = idle, Red = recording, Yellow = transcribing, Green = success.
echo.
echo  Launch it now without rebooting:
echo     wscript run_whisper_stt.vbs
echo  Or for a debug console:
echo     run_whisper_stt.bat
echo.
echo  Press Ctrl+Shift+; in any text field to start dictating.
echo ============================================================================
echo.
pause
endlocal
