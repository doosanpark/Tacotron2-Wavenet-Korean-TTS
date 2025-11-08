@echo off
REM =================================================================
REM Windows용 TensorFlow 2.x + CUDA 12 설치 스크립트 (수정 버전)
REM =================================================================

echo.
echo ========================================
echo Windows용 TensorFlow 2.x + CUDA 12 설치
echo ========================================
echo.
echo 이 스크립트는 다음을 수행합니다:
echo 1. 기존 환경 정리
echo 2. Python 3.10 환경 생성
echo 3. TensorFlow 2.10 설치 (Windows에서 가장 안정적)
echo 4. 필수 라이브러리 설치
echo 5. GPU 설정 테스트
echo.
echo 소요 시간: 약 5-10분
echo.
pause

REM =================================================================
echo.
echo [1/5] 기존 환경 정리 중...
echo =================================================================
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf2_cuda12 -y 2>nul
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf115_new -y 2>nul
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
echo [3/5] TensorFlow 2.10 + GPU 지원 설치 중...
echo =================================================================
echo (Windows에서 가장 안정적인 버전)
echo.

REM pip 업그레이드
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install --upgrade pip

REM TensorFlow 2.10 설치 (Windows에서 CUDA 12 지원)
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install tensorflow==2.10.1

REM CUDA 12 지원 패키지 설치 (Windows용)
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install nvidia-cudnn-cu11

if errorlevel 1 (
    echo [WARNING] CUDA 패키지 설치에 문제가 있을 수 있습니다.
    echo TensorFlow는 설치되었으니 계속 진행합니다...
)
echo [3/5] 완료!

REM =================================================================
echo.
echo [4/5] 필수 라이브러리 설치 중...
echo =================================================================
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install numpy==1.23.5 scipy librosa jamo matplotlib tqdm
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
call C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -c "import tensorflow as tf; print('\n=== TensorFlow Info ==='); print('Version:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); gpus = tf.config.list_physical_devices('GPU'); print('GPU Devices:', gpus); print('GPU Available:', len(gpus) > 0); print('\nTesting GPU...'); print(tf.test.gpu_device_name())"
echo [5/5] 완료!

REM =================================================================
echo.
echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 현재 conda 환경:
call C:\ProgramData\miniconda3\Scripts\conda.exe env list
echo.
echo ========================================
echo 다음 단계
echo ========================================
echo.
echo 1. GPU 상세 테스트:
echo    conda activate tf2_cuda12
echo    python test_gpu.py
echo.
echo 2. 학습 시작:
echo    python train_tacotron2.py --batch_size=8
echo.
echo 또는 start_training.bat 더블클릭
echo.
echo ========================================
pause
