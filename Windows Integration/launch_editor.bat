@echo off
REM Launch the PromptTestEnv project editor from the project folder.
REM Everything resolves against the current working directory,
REM so it must start from the project root (Projects\, config.json, secrets.json).
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo Error: virtual environment not found at .venv\Scripts\activate.bat
    echo Create it first with: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
prompttestenv editor
