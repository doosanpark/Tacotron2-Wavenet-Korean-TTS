# GPU 완전 비활성화 - 수정 완료 요약

## 수정 일시
2025-11-03 15:30

## 문제
```
CPU->GPU Memcpy failed
```
RTX 4060 Ti와 CUDA 10 간의 호환성 문제로 GPU 사용 시 오류 발생

---

## 수정된 파일들

### 1. train_tacotron2.py
**라인 23**: CUDA 완전 비활성화
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

**라인 206-222**: GPU 감지 코드 완전 제거, CPU 전용 모드로 교체
```python
# GPU 완전 비활성화 - CPU 전용 모드
log('Checking TensorFlow version...')
log('TensorFlow version: %s' % tf.__version__)
log('=' * 80)
log('GPU DISABLED - Running in CPU-only mode')
log('This avoids GPU compatibility issues with RTX 4060 Ti and CUDA 10')
log('=' * 80)

# CPU 전용 설정
sess_config = tf.ConfigProto(
    device_count={'GPU': 0},  # GPU를 0개로 설정하여 완전히 비활성화
    log_device_placement=False,
    allow_soft_placement=True,
    inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0
)
log('CPU-only configuration applied: device_count={GPU: 0}')
```

**라인 300-318**: 오류 메시지에서 "GPU" 제거
```python
log('TRAINING FAILED - Detailed Error Information')
log('Exiting due to exception', slack=True)
```

---

### 2. synthesizer.py
**라인 72-76**: CPU 전용 설정 추가
```python
sess_config = tf.ConfigProto(
        device_count={'GPU': 0},  # CPU 전용 모드
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=2)
```

---

### 3. generate.py
**라인 123-124**: CPU 전용 설정 추가
```python
sess_config = tf.ConfigProto(device_count={'GPU': 0})  # CPU 전용
sess = tf.Session(config=sess_config)
```

---

### 4. train_vocoder.py
**라인 208**: CPU 전용 설정 추가
```python
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=False))
```

---

## 적용된 GPU 비활성화 방법

### 1. 환경 변수 레벨
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```
- TensorFlow가 GPU를 인식하지 못하도록 설정

### 2. TensorFlow ConfigProto 레벨
```python
device_count={'GPU': 0}
```
- TensorFlow 세션에서 GPU 디바이스 수를 0으로 강제 설정
- 가장 확실한 GPU 비활성화 방법

---

## 확인사항

### 실행 전 확인
- ✅ CUDA_VISIBLE_DEVICES = '' 설정됨
- ✅ 모든 tf.Session에 device_count={'GPU': 0} 적용됨
- ✅ GPU 감지 코드 제거됨
- ✅ 배치 사이즈 2로 설정됨

### 실행 시 예상 로그
```
TensorFlow version: 1.15.0
================================================================================
GPU DISABLED - Running in CPU-only mode
This avoids GPU compatibility issues with RTX 4060 Ti and CUDA 10
================================================================================
CPU-only configuration applied: device_count={GPU: 0}
```

---

## 실행 방법

### 명령 프롬프트에서:
```bat
cd C:\Users\erid3\Documents\workspace\Tacotron2-Wavenet-Korean-TTS
run_training.bat
```

또는:
```bat
train.bat
```

---

## 예상 효과

### 장점
- ✅ GPU 오류 완전 해결
- ✅ CPU->GPU Memcpy failed 오류 없음
- ✅ 안정적인 훈련 진행

### 단점
- ⚠️ 훈련 속도 느림 (GPU 대비 10-50배)
- ⚠️ 배치 사이즈 2로 제한됨

---

## 주의사항

1. **절대로 GPU 관련 코드 수정하지 말 것**
   - 현재 설정이 최적의 안정성 제공

2. **배치 사이즈 증가 금지**
   - CPU 메모리 한계로 인해 2가 최적값

3. **환경 변수 변경 금지**
   - CUDA_VISIBLE_DEVICES는 빈 문자열로 유지

---

## 롤백 방법 (GPU로 복구하려면)

혹시 나중에 GPU를 다시 사용하고 싶다면:

### 1. train_tacotron2.py:23 수정
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 이 줄 주석 처리
```

### 2. train_tacotron2.py:216 수정
```python
device_count={'GPU': 0},  # 이 줄 제거
```

### 3. 다른 파일들도 동일하게 `device_count={'GPU': 0}` 제거

---

## 문의

추가 문제 발생 시 이 파일 참고하여 수정 내용 확인하세요.
