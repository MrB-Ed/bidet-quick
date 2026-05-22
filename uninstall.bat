@echo off
REM ============================================================================
REM Bidet Quick — uninstaller
REM
REM Removes the Startup shortcut and (optionally) deletes the .venv directory.
REM Does NOT delete the model cache (models/) or your corpus (~/whisper_corpus).
REM ============================================================================

setlocal
cd /d "%~dp0"

echo.
echo === Bidet Quick uninstaller ===
echo.

REM ---- 1. Stop any running instance ------------------------------------------
echo Stopping any running Bidet Quick processes ...
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq Bidet*" >nul 2>&1
REM (Above filter rarely matches because pythonw has no window title; we don't
REM blanket-kill every pythonw.exe because that could hit unrelated tools.)

REM ---- 2. Remove the Startup shortcut ----------------------------------------
set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
if exist "%STARTUP_DIR%\Bidet Quick.vbs" (
    del "%STARTUP_DIR%\Bidet Quick.vbs"
    echo Removed Startup shortcut.
) else (
    echo No Startup shortcut found (already removed?).
)

REM ---- 3. Optionally delete the venv -----------------------------------------
echo.
set /p DELVENV="Delete the .venv directory too? [y/N]: "
if /I "%DELVENV%"=="y" (
    if exist ".venv" (
        rmdir /S /Q ".venv"
        echo .venv deleted.
    ) else (
        echo No .venv to delete.
    )
) else (
    echo Keeping .venv as-is.
)

echo.
echo Done. To fully purge:
echo   - The model cache lives in:  %~dp0models
echo   - Your voice corpus lives in: %USERPROFILE%\whisper_corpus
echo   - Logs live in:               %~dp0logs
echo Delete those manually if you also want them gone.
echo.
pause
endlocal
