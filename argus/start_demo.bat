@echo off
rem ARGUS demo launcher for Windows — double-click or run from a terminal.
rem Delegates everything to run_demo.py (installs deps, seeds, starts the stack).
title ARGUS Demo
cd /d "%~dp0"

if "%~1"=="" (
    python run_demo.py
) else (
    python run_demo.py %*
)

echo.
echo (window can be closed after stopping the demo)
pause
