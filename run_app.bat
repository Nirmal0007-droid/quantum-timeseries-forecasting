@echo off
cd /d "%~dp0"
set "PY="

REM 1) look for a project environment that already has everything installed
for %%P in (".venv\Scripts\python.exe" "venv\Scripts\python.exe") do (
    if not defined PY if exist %%P (
        %%P -c "import streamlit, plotly, torch, pennylane" >nul 2>&1 && set "PY=%%~fP"
    )
)

REM 2) otherwise try the normal system Python
if not defined PY (
    python -c "import streamlit, plotly, torch, pennylane" >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo.
    echo Could not find a Python that has streamlit, plotly, torch and pennylane.
    echo Open the VS Code terminal in this folder and run:
    echo     python -m pip install streamlit plotly
    echo Then double-click this file again.
    echo.
    pause
    exit /b 1
)

echo Using Python: %PY%
"%PY%" -m streamlit run app.py
echo.
echo The app has stopped. Scroll up to read any error message.
pause