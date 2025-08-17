
@echo off
title Create Python venv and install requirements
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_windows.txt
echo.
echo Ready. Use start_cpu.bat or start_gpu.bat
pause
