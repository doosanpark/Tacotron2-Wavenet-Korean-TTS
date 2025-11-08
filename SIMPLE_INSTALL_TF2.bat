@echo off
REM =================================================================
REM 가장 간단한 TensorFlow 2.x 설치 (Windows 최적화)
REM =================================================================

echo.
echo ========================================
echo TensorFlow 2.10 간단 설치 (권장)
echo ========================================
echo.

REM 환경 정리
echo [1/4] 환경 정리...
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf2_gpu -y 2>nul

REM 환경 생성
echo [2/4] Python 3.9 환경 생성... (TensorFlow 2.10 최적)
call C:\ProgramData\miniconda3\Scripts\conda.exe create -n tf2_gpu python=3.9 -y

REM TensorFlow 설치
echo [3/4] TensorFlow 2.10 설치... (Windows에서 가장 안정적)
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -m pip install tensorflow==2.10.1
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -m pip install numpy==1.23.5 scipy librosa jamo matplotlib tqdm

REM 테스트
echo [4/4] GPU 테스트...
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

echo.
echo ========================================
echo 완료! 다음 명령으로 학습 시작:
echo conda activate tf2_gpu
echo python train_tacotron2.py --batch_size=8
echo ========================================
pause
