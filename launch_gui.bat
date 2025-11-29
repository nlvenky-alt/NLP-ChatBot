@echo off
REM ------------------------------------------------------------------
REM 25th ITMC Chatbot - GUI Launcher (no console)
REM Place this file in the project root (next to launch.bat, install.bat).
REM ------------------------------------------------------------------

set "BASE=%~dp0"

REM Prefer local Python from offline installer
set "PYTHON_EXE=%BASE%Python310\pythonw.exe"
if not exist "%PYTHON_EXE%" (
    REM Fallback to system pythonw if available on PATH
    set "PYTHON_EXE=pythonw.exe"
)

REM Offline / embedding environment (same as launch.bat)
set "HF_HOME=%BASE%embedding_models"
set "TRANSFORMERS_CACHE=%BASE%embedding_models"
set "HF_DATASETS_CACHE=%BASE%embedding_models"
set "SENTENCE_TRANSFORMERS_HOME=%BASE%embedding_models"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "TF_ENABLE_ONEDNN_OPTS=0"
set "TF_CPP_MIN_LOG_LEVEL=3"

REM Start Ollama server if not already running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    REM Ollama is already running
) else (
    start /B ollama serve >nul 2>&1
    timeout /t 5 /nobreak >nul
)

REM Launch GUI from the app folder without a console window
cd /d "%BASE%app"
start "" "%PYTHON_EXE%" gui.py
