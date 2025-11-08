# pip 명령어 사용법 (Conda 환경)

## 문제 상황
Windows에서 `pip` 명령어가 실행되지 않는 경우, Python이 PATH에 없거나 Conda 환경이 활성화되지 않았을 수 있습니다.

## 해결 방법

### 방법 1: Conda 환경 활성화 후 사용 (권장)

```batch
REM Conda 환경 활성화
call C:\ProgramData\miniconda3\Scripts\activate.bat tf2_cuda12

REM 이제 pip 사용 가능
python -m pip install -r requirements.txt
```

또는 자동 스크립트 사용:
```batch
install_requirements.bat
```

### 방법 2: 전체 경로로 실행

Conda 환경이 활성화되지 않은 경우:

```batch
REM 전체 경로로 Python 실행
C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -m pip install -r requirements.txt
```

### 방법 3: Conda로 직접 설치

```batch
REM Conda로 직접 설치
conda install tensorflow-gpu=2.13.0 -c conda-forge
conda install numpy scipy matplotlib -c conda-forge
pip install tensorflow-addons librosa jamo tqdm unidecode inflect
```

## 자동 설치 스크립트

프로젝트 루트에 `install_requirements.bat` 파일을 실행하세요:

```batch
install_requirements.bat
```

이 스크립트는:
1. Conda 환경을 자동으로 찾아 활성화
2. pip 업그레이드
3. requirements.txt의 모든 패키지 설치

## 수동 설치 (환경 활성화 후)

Conda 환경을 활성화한 후:

```batch
# 1. pip 업그레이드
python -m pip install --upgrade pip

# 2. TensorFlow 2.x + CUDA 12
python -m pip install tensorflow>=2.13.0 tensorflow-addons>=0.19.0

# 3. 필수 라이브러리
python -m pip install numpy>=1.21.0,<2.0.0 scipy>=1.7.0 librosa>=0.9.0 matplotlib>=3.5.0 jamo>=0.4.1 tqdm>=4.64.0 unidecode>=1.3.0 inflect>=5.0.0
```

또는 requirements.txt 사용:
```batch
python -m pip install -r requirements.txt
```

## 확인 방법

설치가 완료되었는지 확인:

```batch
python -m pip list | findstr /i "tensorflow numpy scipy librosa"
```

또는 Python으로 확인:
```batch
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
```

## 문제 해결

### "python이 인식되지 않습니다"
- Conda 환경을 활성화하세요: `conda activate tf2_cuda12`
- 또는 전체 경로 사용: `C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe`

### "pip이 인식되지 않습니다"
- `python -m pip` 형태로 사용하세요 (권장)
- 또는 `python -m ensurepip --upgrade`로 pip 재설치

### "권한 오류"
- 관리자 권한으로 실행하거나
- `--user` 옵션 사용: `python -m pip install --user -r requirements.txt`

