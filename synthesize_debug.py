# coding: utf-8
import os
import sys
import numpy as np

from text import text_to_sequence, sequence_to_text
from datasets.datafeeder_tacotron2 import _prepare_inputs
from synthesizer import Synthesizer

# 테스트 텍스트
test_texts = [
    "안녕하세요",
    "만나서 반갑습니다"
]

print("=== 시퀀스 변환 디버깅 ===\n")

for i, text in enumerate(test_texts, 1):
    print(f"\n테스트 {i}: {text}")

    # 텍스트를 시퀀스로 변환
    sequence = text_to_sequence(text)
    print(f"  원본 시퀀스 길이: {len(sequence)}")
    print(f"  원본 시퀀스: {sequence}")

    # _prepare_inputs 적용
    prepared = _prepare_inputs([sequence])
    print(f"  Prepared shape: {prepared.shape}")
    print(f"  Prepared 시퀀스: {prepared[0]}")

print("\n\n=== 실제 합성 테스트 ===\n")

# 모델 로드
load_path = "logdir-tacotron2/moon+son_2025-11-09_02-06-17"
sample_path = "logdir-tacotron2/debug_samples"
os.makedirs(sample_path, exist_ok=True)

synthesizer = Synthesizer()
print("모델 로딩 중...")
synthesizer.load(load_path, num_speakers=2, checkpoint_step=40000, inference_prenet_dropout=False)

# synthesizer.py의 synthesize 함수 내부를 확인하기 위해
# 직접 시퀀스 변환 과정을 실행
text = "안녕하세요"
print(f"\n입력 텍스트: {text}")

sequences = np.array([text_to_sequence(text)])
print(f"변환된 sequences shape: {sequences.shape}")
print(f"변환된 sequences: {sequences}")

sequences = _prepare_inputs(sequences)
print(f"Prepared sequences shape: {sequences.shape}")
print(f"Prepared sequences: {sequences}")

input_lengths = [np.argmax(a==1)+1 for a in sequences]
print(f"input_lengths: {input_lengths}")

# 실제 합성
print("\n음성 합성 중...")
audio = synthesizer.synthesize(
    texts=[text],
    base_path=sample_path,
    speaker_ids=[0],
    attention_trim=True,
    isKorean=True
)[0]
print("완료!")
