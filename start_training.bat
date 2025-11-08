@echo off
REM =================================================================
REM 학습 시작 스크립트
REM =================================================================

echo.
echo ========================================
echo Tacotron2 학습 시작
echo ========================================
echo.

REM 환경 활성화
call C:\ProgramData\miniconda3\Scripts\activate.bat tf2_cuda12

REM GPU 확인
echo GPU 상태 확인 중...
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU Found:', len(gpus) > 0, '- Count:', len(gpus))"

echo.
echo 학습을 시작합니다...
echo Batch size: 8
echo.

REM 학습 시작
python train_tacotron2.py --batch_size=8

pause
