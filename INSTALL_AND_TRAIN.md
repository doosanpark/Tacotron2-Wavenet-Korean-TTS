# TensorFlow 2.x + CUDA 12 설치 및 학습 가이드

## 1. 라이브러리 설치

### Windows 환경
```bash
# Python이 설치되어 있는지 확인
python --version

# pip 업그레이드
python -m pip install --upgrade pip

# 필수 라이브러리 설치
python -m pip install -r requirements.txt
```

### Conda 환경 사용 시
```bash
# Conda 환경 활성화 후
conda install tensorflow-gpu=2.13.0 cudatoolkit=12.0 -c conda-forge
pip install -r requirements.txt
```

## 2. CUDA 12 확인

CUDA 12가 설치되어 있는지 확인:
```bash
nvcc --version
```

## 3. GPU 확인

GPU가 제대로 인식되는지 확인:
```bash
python -c "import tensorflow.compat.v1 as tf; tf.disable_v2_behavior(); print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU Found:', len(gpus) > 0)"
```

## 4. 학습 실행

### 방법 1: 배치 파일 사용
```bash
train_tf2_cuda12.bat
```

### 방법 2: 직접 실행
```bash
python train_tacotron2.py --batch_size=4
```

## 5. 주요 변경 사항

- TensorFlow 1.x → TensorFlow 2.x (tf.compat.v1 사용)
- 모든 `tf.log()` → `tf.math.log()`
- 모든 `tf.random_uniform()` → `tf.random.uniform()`
- `tf.scatter_update()` → `Variable.assign()` (wavenet/model.py)
- GPU 메모리 증가 허용 설정 추가

## 6. 문제 해결

### CUDA 오류 발생 시
- CUDA 12와 cuDNN이 설치되어 있는지 확인
- TensorFlow 2.13.0 이상 버전 사용 확인

### 메모리 부족 오류
- `train_tacotron2.py`에서 `--batch_size`를 줄이기 (예: 2 또는 1)

### Import 오류
- tensorflow-addons 설치 확인: `pip install tensorflow-addons`

