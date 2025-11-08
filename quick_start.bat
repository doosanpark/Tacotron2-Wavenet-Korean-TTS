@echo off
REM =================================================================
REM 빠른 시작 스크립트 - 설정 확인 및 학습 실행
REM =================================================================

echo.
echo ========================================
echo Tacotron2-Wavenet Korean TTS
echo TensorFlow 2.x + CUDA 12 빠른 시작
echo ========================================
echo.

REM Conda 환경 활성화
echo Conda 환경 활성화 중...
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call C:\ProgramData\miniconda3\Scripts\activate.bat tf2_cuda12
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call %USERPROFILE%\miniconda3\Scripts\activate.bat tf2_cuda12
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call %USERPROFILE%\anaconda3\Scripts\activate.bat tf2_cuda12
) else (
    echo [경고] Conda 환경을 자동으로 찾을 수 없습니다.
    echo 수동으로 환경을 활성화하세요: conda activate tf2_cuda12
    echo.
)

REM 1. 설정 테스트
echo [1/2] TensorFlow 2.x + CUDA 12 설정 확인 중...
python test_tf2_setup.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 설정 테스트 실패. 오류를 확인하세요.
    pause
    exit /b 1
)

echo.
echo [2/2] 학습 시작...
echo.

REM 2. 학습 실행
python train_tacotron2.py --batch_size=4

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo 학습 중 오류가 발생했습니다.
    echo ========================================
    echo.
    echo 일반적인 해결 방법:
    echo 1. requirements.txt의 라이브러리가 모두 설치되었는지 확인
    echo 2. CUDA 12와 cuDNN이 설치되어 있는지 확인
    echo 3. GPU 드라이버가 최신인지 확인
    echo 4. 배치 크기를 줄여서 다시 시도 (--batch_size=2 또는 1)
    echo.
    pause
    exit /b 1
)

echo.
echo 학습이 완료되었습니다!
pause

