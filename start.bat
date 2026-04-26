@echo off
echo ========================================
echo   StatLab — Starting Backend Server
echo ========================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt -q

echo.
echo Server starting at: http://localhost:5000
echo Open your browser and go to: index.html
echo Press Ctrl+C to stop the server
echo ========================================
cd /d "%~dp0"
python analysis.py
pause
