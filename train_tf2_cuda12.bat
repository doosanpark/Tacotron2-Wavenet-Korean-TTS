@echo off
REM =================================================================
REM TensorFlow 2.x + CUDA 12 학습 시작 스크립트
REM =================================================================

echo.
echo ========================================
echo Tacotron2 학습 시작 (TF2 + CUDA 12)
echo ========================================
echo.

REM Conda 환경 활성화
echo Conda 환경 활성화 중...
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call C:\ProgramData\miniconda3\Scripts\activate.bat tf2_cuda12
    echo   ✓ tf2_cuda12 환경 활성화 완료
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call %USERPROFILE%\miniconda3\Scripts\activate.bat tf2_cuda12
    echo   ✓ tf2_cuda12 환경 활성화 완료
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call %USERPROFILE%\anaconda3\Scripts\activate.bat tf2_cuda12
    echo   ✓ tf2_cuda12 환경 활성화 완료
) else (
    echo   ⚠ Conda 환경을 자동으로 찾을 수 없습니다.
    echo   수동으로 환경을 활성화하세요: conda activate tf2_cuda12
    echo.
)

REM Python 경로 확인
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [오류] Python을 찾을 수 없습니다.
    echo Conda 환경이 활성화되었는지 확인하세요.
    echo.
    echo 해결 방법:
    echo 1. conda activate tf2_cuda12
    echo 2. 또는 전체 경로 사용: C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe
    pause
    exit /b 1
)

REM GPU 확인
echo GPU 상태 확인 중...
python -c "import tensorflow.compat.v1 as tf; tf.disable_v2_behavior(); gpus = tf.config.list_physical_devices('GPU'); print('GPU Found:', len(gpus) > 0, '- Count:', len(gpus))" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo TensorFlow가 설치되어 있지 않거나 오류가 발생했습니다.
    echo requirements.txt에서 라이브러리를 설치하세요: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo 학습을 시작합니다...
echo Batch size: 4 (기본값)
echo 데이터 경로: .\data\moon,.\data\son
echo.

REM 학습 시작
python train_tacotron2.py --batch_size=4

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo 학습 중 오류가 발생했습니다.
    echo ========================================
    pause
    exit /b 1
)

echo.
echo 학습이 완료되었습니다.
pause

