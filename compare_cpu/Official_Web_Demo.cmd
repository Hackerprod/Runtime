@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Launch the upstream Streamlit demo unchanged, bound to localhost only.
set "REPO=%~dp0.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
if not defined PYTHON set "PYTHON=%REPO%\.venv-web-demo\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Python executable not found: "%PYTHON%" 1>&2
  exit /b 2
)
if not exist "%REPO%\scripts\web_demo.py" (
  echo Upstream demo script not found: "%REPO%\scripts\web_demo.py" 1>&2
  exit /b 2
)
if not exist "%REPO%\scripts\minimind-3\config.json" (
  echo Missing model view: "%REPO%\scripts\minimind-3" 1>&2
  echo Create a junction to checkpoints\minimind-3-hf before launching. 1>&2
  exit /b 3
)

"%PYTHON%" -c "import importlib.util,sys; raise SystemExit(0 if importlib.util.find_spec('streamlit') else 4)"
if errorlevel 4 (
  echo Streamlit is not installed in "%PYTHON%". Install it only after review: 1>&2
  echo "%PYTHON%" -m pip install streamlit 1>&2
  exit /b 4
)
if errorlevel 1 exit /b 4

set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
pushd "%REPO%\scripts" || (
  echo Unable to enter upstream scripts directory: "%REPO%\scripts" 1>&2
  exit /b 2
)
echo Official MiniMind Web Demo: http://127.0.0.1:8501
"%PYTHON%" -m streamlit run web_demo.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
