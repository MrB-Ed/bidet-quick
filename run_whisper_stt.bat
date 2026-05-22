@echo off
REM run_whisper_stt.bat - manual launch with visible console for debugging.
REM Uses the repo's .venv if present, else falls back to system python.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" whisper_voice_type.py
) else (
    python whisper_voice_type.py
)

pause
