@echo off
REM =================================================================
REM 라이브러리 설치 스크립트 (Conda 환경 사용)
REM =================================================================

echo.
echo ========================================
echo 라이브러리 설치 시작
echo ========================================
echo.

REM Conda 환경 활성화
echo [1/2] Conda 환경 활성화 중...
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
    echo   ✗ Conda 환경을 찾을 수 없습니다.
    echo.
    echo   다음 중 하나를 시도하세요:
    echo   1. Conda가 설치되어 있는지 확인
    echo   2. 수동으로 환경 활성화: conda activate tf2_cuda12
    echo   3. 전체 경로로 설치:
    echo      C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [2/2] 라이브러리 설치 중...
echo (이 과정은 몇 분이 걸릴 수 있습니다)
echo.

REM pip 업그레이드
python -m pip install --upgrade pip

REM requirements.txt 설치
python -m pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo 설치 중 오류가 발생했습니다.
    echo ========================================
    echo.
    echo 다음을 시도해보세요:
    echo 1. Conda 환경이 활성화되었는지 확인
    echo 2. 인터넷 연결 확인
    echo 3. 수동 설치:
    echo    python -m pip install tensorflow>=2.13.0
    echo    python -m pip install tensorflow-addons>=0.19.0
    echo    python -m pip install numpy scipy librosa matplotlib jamo tqdm unidecode inflect
    pause
    exit /b 1
)

echo.
echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 설치된 패키지 확인:
python -m pip list | findstr /i "tensorflow numpy scipy librosa"
echo.
pause

