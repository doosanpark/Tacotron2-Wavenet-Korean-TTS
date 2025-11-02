#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to revert TensorFlow 2.x compat.v1 imports back to TensorFlow 1.x
"""

import os
import re

files_to_fix = [
    'hparams.py',
    'utils/audio.py',
    'train_tacotron2.py',
    'wavenet/ops.py',
    'wavenet/model.py',
    'wavenet/mixture.py',
    'utils/__init__.py',
    'tacotron2/tacotron2.py',
    'tacotron2/rnn_wrappers.py',
    'tacotron2/modules.py',
    'tacotron2/helpers.py',
    'datasets/datafeeder_wavenet.py',
    'datasets/datafeeder_tacotron2.py',
    'train_vocoder.py',
    'synthesizer.py',
    'generate.py',
]

def revert_tensorflow_import(content):
    """Revert tensorflow.compat.v1 back to tensorflow"""
    # Remove tf.disable_v2_behavior() lines
    content = re.sub(
        r'^tf\.disable_v2_behavior\(\)\n',
        '',
        content,
        flags=re.MULTILINE
    )

    # Replace import tensorflow.compat.v1 as tf with import tensorflow as tf
    content = re.sub(
        r'^import tensorflow\.compat\.v1 as tf$',
        'import tensorflow as tf',
        content,
        flags=re.MULTILINE
    )

    return content

def main():
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path} - file not found")
            continue

        print(f"Processing {file_path}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if needs reverting
        if 'tensorflow.compat.v1' not in content:
            print(f"  Already reverted, skipping...")
            continue

        new_content = revert_tensorflow_import(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  Reverted!")

    print("\nAll files reverted!")

if __name__ == '__main__':
    main()
