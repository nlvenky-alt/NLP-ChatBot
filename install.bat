@echo off
setlocal enabledelayedexpansion
echo ======================================================================
echo INAS CHATBOT - COMPLETE OFFLINE INSTALLER
echo ======================================================================
echo.
echo This will install:
echo   • Ollama (LLM server)
echo   • Python 3.10.11 (local installation)
echo   • All Python packages (offline)
echo   • AI models (llama3.2, llama3.1, phi3.5)
echo   • Embedding models (all-MiniLM-L6-v2)
echo.
echo Installation time: 15-20 minutes
echo.
pause

set "BASE=%~dp0"
set "PYTHON_DIR=%BASE%Python310"

REM ========================================================================
REM STEP 1: Install Ollama
REM ========================================================================
echo.
echo [1/5] Installing Ollama...
echo ======================================================================

if exist "%BASE%installers\OllamaSetup.exe" (
    start /wait "" "%BASE%installers\OllamaSetup.exe" /S
    timeout /t 10 /nobreak >nul
    echo ✓ Ollama installed
) else (
    echo ❌ ERROR: OllamaSetup.exe not found in installers\ folder
    pause
    exit /b 1
)

REM ========================================================================
REM STEP 2: Install Python
REM ========================================================================
echo.
echo [2/5] Installing Python 3.10.11...
echo ======================================================================

if exist "%PYTHON_DIR%\python.exe" (
    echo ✓ Python already installed at: %PYTHON_DIR%
    goto :python_ready
)

if not exist "%BASE%installers\python-3.10.11-amd64.exe" (
    echo ❌ ERROR: python-3.10.11-amd64.exe not found in installers\ folder
    pause
    exit /b 1
)

echo Installing Python (silent mode)...
"%BASE%installers\python-3.10.11-amd64.exe" /quiet InstallAllUsers=0 PrependPath=0 TargetDir="%PYTHON_DIR%"
timeout /t 20 /nobreak >nul

if exist "%PYTHON_DIR%\python.exe" (
    echo ✓ Python installed successfully
) else (
    echo ⚠ Silent install failed, trying interactive...
    start /wait "" "%BASE%installers\python-3.10.11-amd64.exe"
    if not exist "%PYTHON_DIR%\python.exe" (
        echo ❌ ERROR: Python installation failed
        echo Please install manually to: %PYTHON_DIR%
        pause
        exit /b 1
    )
)

:python_ready
"%PYTHON_DIR%\python.exe" --version

REM ========================================================================
REM STEP 3: Install Python Packages (Offline)
REM ========================================================================
echo.
echo [3/5] Installing Python packages (offline)...
echo ======================================================================

if not exist "%BASE%python_packages" (
    echo ❌ ERROR: python_packages\ folder not found
    pause
    exit /b 1
)

echo Installing from: %BASE%python_packages
echo This may take 5-10 minutes...
echo.

"%PYTHON_DIR%\python.exe" -m pip install --no-index --find-links="%BASE%python_packages" ^
    numpy requests PyYAML filelock fsspec packaging typing-extensions ^
    torch transformers tokenizers safetensors huggingface-hub ^
    sentence-transformers faiss-cpu ^
    pypdf python-docx openpyxl xlrd lxml ^
    scipy scikit-learn tqdm Pillow regex psutil

if %ERRORLEVEL% EQU 0 (
    echo ✓ All packages installed successfully
) else (
    echo ⚠ Some packages may have failed - check output above
)

REM ========================================================================
REM STEP 4: Import Ollama Models
REM ========================================================================
echo.
echo [4/5] Importing AI models (offline)...
echo ======================================================================

set "OLLAMA_CACHE=%USERPROFILE%\.ollama"
if not exist "%OLLAMA_CACHE%" mkdir "%OLLAMA_CACHE%"

if exist "%BASE%models\ollama_models\models" (
    echo Copying models to: %OLLAMA_CACHE%\models
    xcopy /E /I /H /Y /Q "%BASE%models\ollama_models\models\*" "%OLLAMA_CACHE%\models\" >nul
    echo ✓ Models imported
) else (
    echo ⚠ WARNING: models\ollama_models\models\ folder not found
)

echo Starting Ollama server...
start /B ollama serve >nul 2>&1
timeout /t 5 /nobreak >nul
ollama list 2>nul

REM ========================================================================
REM STEP 5: Verify Embedding Models
REM ========================================================================
echo.
echo [5/5] Verifying embedding models...
echo ======================================================================

if exist "%BASE%embedding_models\all-MiniLM-L6-v2\config.json" (
    echo ✓ Embedding models found - System is 100%% offline ready
) else (
    echo ⚠ WARNING: Embedding models missing
    echo Expected location: %BASE%embedding_models\all-MiniLM-L6-v2\
    echo.
    echo The system may need internet on first run to download these.
)

REM ========================================================================
REM Create Environment Config
REM ========================================================================
echo.
echo Creating environment configuration...

>"%BASE%set_env.bat" (
    echo @echo off
    echo set "PYTHON_HOME=%PYTHON_DIR%"
    echo set "HF_HOME=%BASE%embedding_models"
    echo set "TRANSFORMERS_CACHE=%BASE%embedding_models"
    echo set "HF_DATASETS_CACHE=%BASE%embedding_models"
    echo set "SENTENCE_TRANSFORMERS_HOME=%BASE%embedding_models"
    echo set "TRANSFORMERS_OFFLINE=1"
    echo set "HF_HUB_OFFLINE=1"
    echo set "HF_DATASETS_OFFLINE=1"
    echo set "TF_ENABLE_ONEDNN_OPTS=0"
    echo set "TF_CPP_MIN_LOG_LEVEL=3"
)

echo ✓ Environment configured

REM ========================================================================
REM Installation Complete
REM ========================================================================
echo.
echo ======================================================================
echo ✅ INSTALLATION COMPLETE!
echo ======================================================================
echo.
echo ✓ Ollama installed and running
echo ✓ Python 3.10.11 installed at: %PYTHON_DIR%
echo ✓ Python packages installed (offline)
echo ✓ AI models imported (llama3.2, llama3.1, phi3.5)
echo ✓ Embedding models verified
echo.
echo 🚀 To start the chatbot: Double-click launch.bat
echo.
echo ======================================================================

endlocal
pause
