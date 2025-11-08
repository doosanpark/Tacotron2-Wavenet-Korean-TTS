@echo off
REM =================================================================
REM Tacotron2 학습 시작 (GPU 수정 버전)
REM =================================================================

echo.
echo ========================================
echo Tacotron2 GPU 학습 시작!
echo ========================================
echo.

REM CUDA 경로 설정
set PATH=C:\ProgramData\miniconda3\envs\tf2_gpu\Library\bin;%PATH%

REM GPU 확인
echo GPU 상태 확인...
C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU Available:', len(gpus) > 0); print('GPU Count:', len(gpus))"

if errorlevel 1 (
    echo.
    echo [ERROR] GPU를 확인할 수 없습니다!
    echo FIX_GPU_CONDA.bat를 먼저 실행하세요.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 학습 설정
echo ========================================
echo - Environment: tf2_gpu
echo - Batch Size: 8
echo - GPU: Enabled
echo ========================================
echo.

timeout /t 3 /nobreak

REM 학습 시작
C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe train_tacotron2.py --batch_size=8

pause
