@echo off
echo ========================================
echo [Step 1/5] 기존 TensorFlow 1.x 환경 제거
echo ========================================
echo.

REM 기존 환경 제거
echo 기존 tf115_new 환경을 제거합니다...
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf115_new -y

echo.
echo 현재 conda 환경 목록:
call C:\ProgramData\miniconda3\Scripts\conda.exe env list

echo.
echo ========================================
echo Step 1 완료!
echo ========================================
pause
