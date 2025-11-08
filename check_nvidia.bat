@echo off
echo ========================================
echo NVIDIA GPU 상태 확인
echo ========================================
echo.

echo [1] nvidia-smi 실행 (GPU 하드웨어 확인)
echo ========================================
nvidia-smi
echo.

echo [2] CUDA 경로 확인
echo ========================================
echo CUDA_PATH: %CUDA_PATH%
echo CUDA_PATH_V12_X: %CUDA_PATH_V12_0% %CUDA_PATH_V12_1% %CUDA_PATH_V12_2%
echo.

echo [3] TensorFlow 빌드 정보
echo ========================================
C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('CUDA Version (built):', tf.version.CUDA if hasattr(tf.version, 'CUDA') else 'N/A'); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"
echo.

echo [4] 상세 진단
echo ========================================
C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -c "import os; print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'Not set')); print('PATH에 CUDA 있나:', 'cuda' in os.environ.get('PATH', '').lower())"
echo.

pause
