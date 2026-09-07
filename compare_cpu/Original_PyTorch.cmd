@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PY=%MINIMIND_PYTHON%"
if not defined PY set "PY=C:\Python314\python.exe"
if not exist "%PY%" if defined MINIMIND_PYTHON goto :missing_python
if not exist "%PY%" set "PY=python"
if /I "%PY%"=="python" (
    where python >nul 2>&1
    if errorlevel 1 goto :missing_python
)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
pushd "%~dp0.." || goto :missing_repo
"%PY%" -m native_cpu.tools.compare_chat --backend original %*
set "RC=%ERRORLEVEL%"
popd
if not "%MINIMIND_NO_PAUSE%"=="1" pause
endlocal & exit /b %RC%
:missing_repo
set "RC=2"
echo Repository directory not found 1>&2
if not "%MINIMIND_NO_PAUSE%"=="1" pause
endlocal & exit /b %RC%
:missing_python
set "RC=2"
echo Python not found; set MINIMIND_PYTHON to an executable 1>&2
if not "%MINIMIND_NO_PAUSE%"=="1" pause
endlocal & exit /b %RC%
