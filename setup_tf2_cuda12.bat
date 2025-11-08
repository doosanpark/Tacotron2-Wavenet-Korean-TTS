@echo off
REM TensorFlow 2.x + CUDA 12 설치 스크립트

echo ========================================
echo TensorFlow 2.x + CUDA 12 설치 시작
echo ========================================

REM 1. 기존 tf115_new 환경 제거
echo.
echo [1/5] 기존 TensorFlow 1.x 환경 제거 중...
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf115_new -y

REM 2. 새로운 환경 생성 (Python 3.10 - TF2와 호환)
echo.
echo [2/5] 새로운 Python 환경 생성 중 (Python 3.10)...
call C:\ProgramData\miniconda3\Scripts\conda.exe create -n tf2_cuda12 python=3.10 -y

REM 3. TensorFlow 2.x + CUDA 12 설치
echo.
echo [3/5] TensorFlow 2.15 + CUDA 12 설치 중...
call C:\ProgramData\miniconda3\Scripts\conda.exe activate tf2_cuda12
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install --upgrade pip
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install tensorflow[and-cuda]==2.15.0

REM 4. 필수 라이브러리 설치
echo.
echo [4/5] 필수 라이브러리 설치 중...
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install numpy scipy librosa jamo matplotlib tqdm

REM 5. 설치 확인
echo.
echo [5/5] 설치 확인 중...
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

echo.
echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 다음 명령으로 학습을 시작하세요:
echo conda activate tf2_cuda12
echo python train_tacotron2.py --batch_size=8
echo.
pause
