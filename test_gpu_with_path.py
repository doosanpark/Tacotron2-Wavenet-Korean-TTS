import os
import sys

# Add conda CUDA libraries to PATH
cuda_path = r'C:\ProgramData\miniconda3\envs\tf2_gpu\Library\bin'
os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')

print('CUDA path added:', cuda_path)
print('=' * 80)

import tensorflow as tf

print('TensorFlow:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', gpus)

if gpus:
    print('\n' + '=' * 80)
    print('SUCCESS! GPU is available and ready to use!')
    print('GPU Name:', gpus[0].name)
    print('=' * 80)

    # Set memory growth to avoid OOM errors
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('GPU memory growth enabled')
else:
    print('\n' + '=' * 80)
    print('WARNING: No GPU detected - will use CPU')
    print('=' * 80)
