#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert TensorFlow 1.x imports to TensorFlow 2.x compat.v1
"""

import os
import re

files_to_fix = [
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

def fix_tensorflow_import(content):
    """Replace tensorflow imports with compat.v1"""
    # Replace import tensorflow as tf
    content = re.sub(
        r'^import tensorflow as tf$',
        'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()',
        content,
        flags=re.MULTILINE
    )

    # Replace import tensorflow
    content = re.sub(
        r'^import tensorflow$',
        'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()',
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

        # Check if already fixed
        if 'tensorflow.compat.v1' in content:
            print(f"  Already fixed, skipping...")
            continue

        new_content = fix_tensorflow_import(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  Fixed!")

    print("\nAll files processed!")

if __name__ == '__main__':
    main()
