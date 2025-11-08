# Tacotron2 훈련 오류 분석 리포트

## 실행 일시
- 생성: 2025-11-03 14:03
- 최근 실패한 훈련: 2025-11-03 10:21-10:25

---

## 문제 요약

### 주요 오류 1: GPU GEMM 연산 실패 (10:21-10:25 실행)

```
Internal: Blas GEMM launch failed : a.shape=(336, 512), b.shape=(512, 128)
[[node model/Decoder/BahdanauMonotonicAttention/memory_layer/Tensordot/MatMul]]
```

**원인:**
- CUDA 버전 불일치
  - 시스템 CUDA: 12.6
  - TensorFlow 1.15 요구사항: CUDA 10.0 또는 10.1
  - GPU: RTX 4060 Ti (8GB)

**증상:**
- TensorFlow 로드 성공 ✓
- GPU 감지 성공 ✓
- 모델 초기화 성공 ✓
- 훈련 시작 후 GPU 행렬 연산에서 실패 ✗

---

### 주요 오류 2: TensorFlow DLL 로드 실패 (이후 실행들)

```
ImportError: DLL load failed: 지정된 모듈을 찾을 수 없습니다.
```

**원인:**
- Python 직접 실행 시 환경 변수 PATH 누락
- 필요한 DLL 파일들:
  1. CUDA DLL: cudart64_100.dll, cublas64_100.dll, cufft64_100.dll
  2. cuDNN DLL: cudnn64_7.dll
  3. MSVC DLL: msvcp140.dll, vcruntime140.dll

---

## 수정된 파일들

### 1. train_tacotron2.py (라인 248)
```python
# 수정 전:
sess_config.gpu_options.visible_device_list = '0'

# 수정 후:
# sess_config.gpu_options.visible_device_list = '0'  # 주석 처리
```

### 2. train.bat
```bat
# 수정 전:
python train_tacotron2.py --load_path logdir-tacotron2/moon+son_2025-11-02_18-45-07

# 수정 후:
python train_tacotron2.py --batch_size=2
```

### 3. run_training.bat
```bat
# 수정 전:
python train_tacotron2.py

# 수정 후:
python train_tacotron2.py --batch_size=2
```

---

## 해결 방안

### 방안 1: CPU 모드로 실행 (권장)

**장점:**
- CUDA 버전 불일치 문제 회피
- 즉시 실행 가능

**단점:**
- 훈련 속도가 매우 느림 (GPU 대비 10-50배 느림)

**실행 방법:**
```bat
cd C:\Users\erid3\Documents\workspace\Tacotron2-Wavenet-Korean-TTS
run_training.bat
```

또는:
```bat
train.bat
```

**현재 상태:**
- GPU 비활성화 코드 적용됨 (train_tacotron2.py:23)
- 배치 사이즈 2로 감소

---

### 방안 2: CUDA Toolkit 10.0 설치

**필요 작업:**

1. CUDA 10.0과 cuDNN 7.6을 conda로 설치:
```bat
conda install -n tf115_new cudatoolkit=10.0 cudnn=7.6 -y
```

2. GPU 재활성화 (train_tacotron2.py:248 주석 해제):
```python
sess_config.gpu_options.visible_device_list = '0'
```

3. 훈련 실행

**장점:**
- GPU 가속으로 빠른 훈련
- 원래 의도한 방식

**단점:**
- CUDA 12.6과 충돌 가능성
- 추가 설정 필요

---

### 방안 3: TensorFlow 2.x로 업그레이드

**필요 작업:**

1. 새 환경 생성:
```bat
conda create -n tf2_gpu python=3.8 tensorflow-gpu=2.10 -y
```

2. 코드 마이그레이션 (대규모 수정 필요)

**장점:**
- CUDA 12.6 호환
- 최신 기능 사용 가능

**단점:**
- TensorFlow 1.x → 2.x 마이그레이션 필요
- 코드 대폭 수정 필요
- 시간이 많이 걸림

---

## 현재 환경 정보

### GPU
- 모델: NVIDIA GeForce RTX 4060 Ti
- 메모리: 8188 MB
- CUDA 버전: 12.6
- Driver: 560.94

### Conda 환경
- 이름: tf115_new
- Python: 3.7
- TensorFlow: 1.15.0
- 경로: C:\ProgramData\miniconda3\envs\tf115_new

### 데이터
- moon: 34 examples (0.02 hours)
- son: 38 examples (0.05 hours)
- 총 72 examples

### 모델
- 파라미터: 29.229 Million
- 배치 사이즈: 2 (수정됨)

---

## 다음 단계 권장사항

### 즉시 실행 (CPU 모드)

명령 프롬프트를 열어서:
```bat
cd C:\Users\erid3\Documents\workspace\Tacotron2-Wavenet-Korean-TTS
run_training.bat
```

**예상 결과:**
- CPU로 훈련 진행
- 느리지만 안정적
- 배치 사이즈 2로 메모리 부담 감소

### 장기 해결 (CUDA 10.0 설치)

1. CUDA Toolkit 10.0 다운로드 및 설치
2. cuDNN 7.6 설치
3. GPU 재활성화
4. 훈련 재시작

---

## 로그 파일 위치

- 최근 실패한 훈련: `logdir-tacotron2/moon+son_2025-11-03_10-21-29/train.log`
- 훈련 출력 로그: `training_output.log`

---

## 문의사항

추가 문의사항이 있으시면 이 리포트를 참고하여 질문해주세요.
