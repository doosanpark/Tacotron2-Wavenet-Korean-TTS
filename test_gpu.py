# coding: utf-8
"""
GPU 설정 테스트 스크립트

TensorFlow 2.x + CUDA 12 환경에서 GPU가 올바르게 인식되고 동작하는지 테스트합니다.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys

print('='*80)
print('GPU 설정 테스트')
print('='*80)

# 1. TensorFlow 버전 확인
print(f'\n[1] TensorFlow 버전: {tf.__version__}')

# 2. GPU 장치 확인 (TF2 API 사용)
print('\n[2] GPU 장치 확인 (TF2 API):')
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f'   - GPU 발견: {len(physical_devices)}개')
        for i, gpu in enumerate(physical_devices):
            print(f'   - GPU {i}: {gpu}')
            try:
                # GPU 메모리 증가 설정
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f'   - GPU {i}: 메모리 증가 설정 완료')
            except Exception as e:
                print(f'   - GPU {i}: 메모리 설정 실패 - {e}')
    else:
        print('   - GPU를 찾을 수 없습니다.')
        print('   - CPU 모드로 실행됩니다.')
except Exception as e:
    print(f'   - GPU 확인 중 오류 발생: {e}')

# 3. Session에서 GPU 확인 (TF1 스타일)
print('\n[3] Session에서 GPU 확인 (TF1 스타일):')
sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True  # 디바이스 배치 로그 표시
)
sess_config.gpu_options.allow_growth = True

with tf.Session(config=sess_config) as sess:
    # 간단한 연산 수행
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)

    print('\n   간단한 행렬 곱셈 테스트:')
    print('   A = [[1, 2], [3, 4]]')
    print('   B = [[5, 6], [7, 8]]')
    result = sess.run(c)
    print(f'   A × B = {result.tolist()}')

# 4. CUDA 버전 확인
print('\n[4] CUDA 버전 확인:')
try:
    from tensorflow.python.platform import build_info
    print(f'   - CUDA 버전: {build_info.cuda_version_number}')
    print(f'   - cuDNN 버전: {build_info.cudnn_version_number}')
except:
    print('   - CUDA 버전 정보를 가져올 수 없습니다.')

# 5. 권장 사항
print('\n[5] 권장 사항:')
if physical_devices:
    print('   ✅ GPU가 올바르게 감지되었습니다!')
    print('   ✅ 학습 스크립트를 실행할 수 있습니다.')
    print('\n   학습 시작 명령어:')
    print('   python train_tacotron2.py --batch_size=4')
else:
    print('   ⚠️ GPU가 감지되지 않았습니다.')
    print('   다음을 확인하세요:')
    print('   1. NVIDIA GPU 드라이버가 설치되어 있는지')
    print('   2. CUDA 12.x가 설치되어 있는지')
    print('   3. cuDNN이 올바르게 설치되어 있는지')
    print('   4. TensorFlow-GPU가 설치되어 있는지')
    print('\n   설치 명령어:')
    print('   pip install tensorflow[and-cuda]==2.15.0')

print('\n'+'='*80)
print('테스트 완료')
print('='*80)
