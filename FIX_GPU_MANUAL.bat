@echo off
REM =================================================================
REM GPU 문제 해결: 수동 설치 (CUDA Toolkit 포함)
REM =================================================================

echo.
echo ========================================
echo GPU 문제 수동 해결
echo ========================================
echo.

echo [1/6] 환경 정리...
call C:\ProgramData\miniconda3\Scripts\conda.exe env remove -n tf2_cuda12 -y

echo [2/6] Python 3.9 환경 생성...
call C:\ProgramData\miniconda3\Scripts\conda.exe create -n tf2_gpu python=3.9 -y

echo [3/6] pip 업그레이드...
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -m pip install --upgrade pip

echo [4/6] TensorFlow-GPU 2.10 설치...
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -m pip install tensorflow-gpu==2.10.1

echo [5/6] CUDA Toolkit 11.2 설치 (conda)...
call C:\ProgramData\miniconda3\Scripts\conda.exe install -n tf2_gpu -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y

echo [6/6] 필수 라이브러리...
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -m pip install numpy==1.23.5 scipy librosa jamo matplotlib tqdm

echo.
echo ========================================
echo GPU 테스트
echo ========================================
call C:\ProgramData\miniconda3\envs\tf2_gpu\python.exe -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

echo.
pause
