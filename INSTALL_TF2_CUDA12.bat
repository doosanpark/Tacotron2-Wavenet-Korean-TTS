@echo off
REM =================================================================
REM TensorFlow 2.15 + CUDA 12 완전 설치 스크립트
REM =================================================================

echo.
echo ========================================
echo TensorFlow 2.x + CUDA 12 설치 시작
echo ========================================
echo.
echo 이 스크립트는 다음을 수행합니다:
echo 1. 기존 TensorFlow 1.x 환경 제거
echo 2. Python 3.10 환경 생성
echo 3. TensorFlow 2.15 + CUDA 12 설치
echo 4. 필수 라이브러리 설치
echo 5. GPU 설정 테스트
echo.
echo 소요 시간: 약 5-10분
echo.
pause

REM =================================================================
echo.
echo [1/5] 기존 환경 제거 중...
echo =================================================================
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf115_new -y
if errorlevel 1 (
    echo [WARNING] tf115_new 환경이 없거나 제거 중 문제가 발생했습니다.
    echo 계속 진행합니다...
)
echo [1/5] 완료!

REM =================================================================
echo.
echo [2/5] 새로운 Python 3.10 환경 생성 중...
echo =================================================================
call C:\ProgramData\miniconda3\Scripts\conda.exe create -n tf2_cuda12 python=3.10 -y
if errorlevel 1 (
    echo [ERROR] 환경 생성 실패!
    pause
    exit /b 1
)
echo [2/5] 완료!

REM =================================================================
echo.
echo [3/5] TensorFlow 2.15 + CUDA 12 설치 중...
echo =================================================================
echo (이 단계가 가장 오래 걸립니다 - 약 3-5분)
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install --upgrade pip
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install tensorflow[and-cuda]==2.15.0
if errorlevel 1 (
    echo [ERROR] TensorFlow 설치 실패!
    pause
    exit /b 1
)
echo [3/5] 완료!

REM =================================================================
echo.
echo [4/5] 필수 라이브러리 설치 중...
echo =================================================================
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install numpy scipy librosa jamo matplotlib tqdm
if errorlevel 1 (
    echo [ERROR] 라이브러리 설치 실패!
    pause
    exit /b 1
)
echo [4/5] 완료!

REM =================================================================
echo.
echo [5/5] GPU 설정 테스트 중...
echo =================================================================
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -c "import tensorflow as tf; print('\n=== TensorFlow Info ==='); print('Version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', gpus); print('GPU Available:', len(gpus) > 0)"
if errorlevel 1 (
    echo [WARNING] GPU 테스트 중 문제가 발생했습니다.
    echo TensorFlow는 설치되었지만 GPU를 사용하지 못할 수 있습니다.
)
echo [5/5] 완료!

REM =================================================================
echo.
echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 현재 conda 환경 목록:
call C:\ProgramData\miniconda3\Scripts\conda.exe env list
echo.
echo ========================================
echo 다음 단계
echo ========================================
echo.
echo 1. 새 터미널을 열고 다음 명령 실행:
echo    conda activate tf2_cuda12
echo.
echo 2. GPU 테스트:
echo    python test_gpu.py
echo.
echo 3. 학습 시작:
echo    python train_tacotron2.py --batch_size=8
echo.
echo ========================================
pause
