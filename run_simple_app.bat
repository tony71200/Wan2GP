@echo off
cd /d "%~dp0"

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" wan2gp
if errorlevel 1 (
    echo [ERROR] Khong activate duoc env.
    pause
    exit /b 1
)

python simple_app.py
pause
