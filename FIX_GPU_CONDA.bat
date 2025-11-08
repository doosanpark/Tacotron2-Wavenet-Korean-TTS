@echo off
REM =================================================================
REM GPU 문제 해결: Conda를 통한 TensorFlow-GPU 설치
REM (Conda가 CUDA/cuDNN을 자동으로 관리합니다)
REM =================================================================

echo.
echo ========================================
echo GPU 문제 해결 시작
echo ========================================
echo.
echo 이 스크립트는:
echo 1. 기존 환경 제거
echo 2. Conda를 통해 TensorFlow-GPU 설치
echo 3. CUDA/cuDNN 자동 설정
echo ========================================
echo.
pause

REM 환경 제거
echo [1/4] 기존 환경 제거...
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf2_cuda12 -y

REM 새 환경 생성
echo [2/4] Python 3.9 환경 생성...
call C:\ProgramData\miniconda3\Scripts\conda.exe create -n tf2_gpu python=3.9 -y

REM Conda로 TensorFlow-GPU 설치 (CUDA 자동 설치)
echo [3/4] Conda로 TensorFlow-GPU + CUDA 설치...
echo (Conda가 필요한 CUDA/cuDNN을 자동으로 설치합니다)
call C:\ProgramData\miniconda3\Scripts\conda.exe install -n tf2_gpu -c conda-forge tensorflow-gpu=2.10 -y

REM 필수 라이브러리
echo [4/4] 필수 라이브러리 설치...
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -m pip install numpy==1.23.5 scipy librosa jamo matplotlib tqdm

REM GPU 테스트
echo.
echo ========================================
echo GPU 테스트
echo ========================================
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -c "import tensorflow as tf; print('\nTensorFlow:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU Found:', len(gpus) > 0); print('GPU Count:', len(gpus)); [print('  GPU:', gpu) for gpu in gpus]"

echo.
echo ========================================
echo 완료!
echo ========================================
echo.
echo 다음 명령으로 학습 시작:
echo conda activate tf2_gpu
echo python train_tacotron2.py --batch_size=8
echo.
pause
