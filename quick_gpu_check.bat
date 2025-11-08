@echo off
C:\ProgramData\miniconda3\envs\tf2_cuda12\python.exe -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU Found:', len(gpus) > 0); print('GPU Count:', len(gpus)); [print('  -', gpu) for gpu in gpus]" 2>&1
