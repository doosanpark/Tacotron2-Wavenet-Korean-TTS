@echo off
REM =================================================================
REM Tacotron2 학습 시작 (TensorFlow 2.x + GPU)
REM =================================================================

echo.
echo ========================================
echo Tacotron2 GPU 학습 시작!
echo ========================================
echo.

REM GPU 상태 빠른 확인
echo GPU 상태 확인 중...
C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU Available:', len(gpus) > 0); print('GPU Count:', len(gpus))"

echo.
echo ========================================
echo 학습 설정
echo ========================================
echo - Batch Size: 8
echo - Data: ./data/moon, ./data/son
echo - GPU: Enabled
echo ========================================
echo.

echo 학습을 시작합니다... (Ctrl+C로 중단 가능)
timeout /t 3 /nobreak

REM 학습 시작
C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe train_tacotron2.py --batch_size=8

echo.
echo ========================================
echo 학습이 종료되었습니다.
echo ========================================
pause
