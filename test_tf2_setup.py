# coding: utf-8
"""
TensorFlow 2.x + CUDA 12 설정 테스트 스크립트
"""
import sys
import os

print("=" * 80)
print("TensorFlow 2.x + CUDA 12 설정 테스트")
print("=" * 80)
print()

# 1. TensorFlow import 테스트
print("[1] TensorFlow import 테스트...")
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print(f"  ✓ TensorFlow 버전: {tf.__version__}")
except Exception as e:
    print(f"  ✗ TensorFlow import 실패: {e}")
    sys.exit(1)

# 2. GPU 확인
print("\n[2] GPU 확인...")
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"  ✓ GPU 발견: {len(physical_devices)}개")
        for i, device in enumerate(physical_devices):
            print(f"    - GPU {i}: {device}")
    else:
        print("  ⚠ GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.")
except Exception as e:
    print(f"  ⚠ GPU 확인 실패: {e}")

# 3. GPU 메모리 설정 테스트
print("\n[3] GPU 메모리 설정 테스트...")
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("  ✓ GPU 메모리 증가 허용 설정 완료")
    else:
        print("  ⚠ GPU가 없어 건너뜀")
except Exception as e:
    print(f"  ⚠ GPU 메모리 설정 실패: {e}")

# 4. TensorFlow 연산 테스트
print("\n[4] TensorFlow 연산 테스트...")
try:
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess_config.gpu_options.allow_growth = True
    
    with tf.Session(config=sess_config) as sess:
        a = tf.constant(2.0)
        b = tf.constant(3.0)
        c = a + b
        result = sess.run(c)
        print(f"  ✓ 간단한 연산 테스트 성공: 2.0 + 3.0 = {result}")
except Exception as e:
    print(f"  ✗ TensorFlow 연산 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 필수 라이브러리 확인
print("\n[5] 필수 라이브러리 확인...")
required_libs = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'librosa': 'librosa',
    'matplotlib': 'matplotlib',
    'jamo': 'jamo',
    'tqdm': 'tqdm',
    'tensorflow_addons': 'tensorflow-addons'
}

missing_libs = []
for lib_name, import_name in required_libs.items():
    try:
        __import__(import_name)
        print(f"  ✓ {lib_name}")
    except ImportError:
        print(f"  ✗ {lib_name} (누락)")
        missing_libs.append(lib_name)

if missing_libs:
    print(f"\n  ⚠ 누락된 라이브러리: {', '.join(missing_libs)}")
    print("  다음 명령어로 설치하세요:")
    print(f"  pip install {' '.join(missing_libs)}")

print("\n" + "=" * 80)
print("테스트 완료!")
if not missing_libs:
    print("모든 설정이 완료되었습니다. 학습을 시작할 수 있습니다.")
else:
    print("누락된 라이브러리를 설치한 후 다시 테스트하세요.")
print("=" * 80)

