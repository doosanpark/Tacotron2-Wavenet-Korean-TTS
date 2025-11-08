import tensorflow as tf

print('TensorFlow:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', gpus)

if gpus:
    print('GPU Name:', gpus[0].name)
    print('GPU is available and ready to use!')
else:
    print('No GPU detected - will use CPU')
