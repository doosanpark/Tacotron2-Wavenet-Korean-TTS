@echo off
echo ========================================
echo GPU 테스트 실행 중...
echo ========================================
echo.

C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe test_gpu.py

echo.
echo ========================================
echo 테스트 완료!
echo ========================================
pause
