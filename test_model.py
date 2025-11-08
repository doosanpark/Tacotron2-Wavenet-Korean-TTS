# -*- coding: utf-8 -*-
import os
os.environ['PATH'] = r'C:\ProgramData\miniconda3\envs\tf2_gpu\Library\bin' + ';' + os.environ.get('PATH', '')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from hparams import hparams
from tacotron2.tacotron2 import Tacotron2

# Create a simple graph to test model construction
with tf.Graph().as_default():
    # Create dummy inputs
    inputs = tf.placeholder(tf.int32, [2, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [2], 'input_lengths')
    mel_targets = tf.placeholder(tf.float32, [2, None, hparams.num_mels], 'mel_targets')
    speaker_id = tf.placeholder(tf.int32, [2], 'speaker_id')

    print("Creating Tacotron2 model...")
    model = Tacotron2(hparams)
    model.initialize(inputs=inputs, input_lengths=input_lengths,
                    mel_targets=mel_targets, speaker_id=speaker_id,
                    num_speakers=2)

    print("Model created successfully!")
