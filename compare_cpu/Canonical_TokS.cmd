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
rem Default canonical guard is CPU-E6; set CANONICAL_LIBRARY and
rem CANONICAL_EXPECTED_SHA256 only for an explicit A/B comparison.
pushd "%~dp0.." || goto :missing_repo
if defined CANONICAL_LIBRARY (
    if not defined CANONICAL_EXPECTED_SHA256 goto :missing_canonical_hash
    if defined CANONICAL_RESULTS (
        "%PY%" compare_cpu\canonical_tok_s.py --library "%CANONICAL_LIBRARY%" --expected-sha256 "%CANONICAL_EXPECTED_SHA256%" --results "%CANONICAL_RESULTS%"
    ) else (
        "%PY%" compare_cpu\canonical_tok_s.py --library "%CANONICAL_LIBRARY%" --expected-sha256 "%CANONICAL_EXPECTED_SHA256%"
    )
) else (
    "%PY%" compare_cpu\canonical_tok_s.py
)
set "RC=%ERRORLEVEL%"
popd
if not "%MINIMIND_NO_PAUSE%"=="1" pause
endlocal & exit /b %RC%

:missing_python
echo Could not find Python. Set MINIMIND_PYTHON to a valid interpreter. 1>&2
endlocal & exit /b 1

:missing_repo
echo Could not open the MiniMind repository. 1>&2
endlocal & exit /b 1

:missing_canonical_hash
echo CANONICAL_EXPECTED_SHA256 is required when CANONICAL_LIBRARY is set. 1>&2
endlocal & exit /b 1
