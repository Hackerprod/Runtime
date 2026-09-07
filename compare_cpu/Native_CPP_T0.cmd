@echo off
setlocal EnableExtensions DisableDelayedExpansion
call "%~dp0Native_CPP.cmd" --threads 2 --cpus 0,6 --row-weights 1332,992 %*
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
