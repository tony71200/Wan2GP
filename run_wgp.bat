@echo off
cd /d "%~dp0"

REM Sua tro toi thu muc goc conda cua ban
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" wan2gp
if errorlevel 1 (
    echo [ERROR] Khong activate duoc env.
    pause
    exit /b 1
)

python wgp.py --profile 4
pause
