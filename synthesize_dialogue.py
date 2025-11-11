# coding: utf-8
import os
import sys

# 텍스트 파일에서 읽기
with open('speaker1_text.txt', 'r', encoding='utf-8') as f:
    speaker1_text = f.read().strip()

with open('speaker2_text.txt', 'r', encoding='utf-8') as f:
    speaker2_text = f.read().strip()

print(f"화자1 텍스트: {speaker1_text}")
print(f"화자2 텍스트: {speaker2_text}")

from synthesizer import Synthesizer

# 모델 로드
load_path = "logdir-tacotron2/moon+son_2025-11-09_02-06-17"
sample_path = "logdir-tacotron2/dialogue_samples"
os.makedirs(sample_path, exist_ok=True)

synthesizer = Synthesizer()
synthesizer.load(load_path, num_speakers=2, checkpoint_step=None, inference_prenet_dropout=False)

# 화자1 음성 합성
print("\n화자1 음성 합성 중...")
audio1 = synthesizer.synthesize(
    texts=[speaker1_text],
    base_path=sample_path,
    speaker_ids=[0],
    attention_trim=True,
    isKorean=True
)[0]
print("화자1 완료!")

# 화자2 음성 합성
print("\n화자2 음성 합성 중...")
audio2 = synthesizer.synthesize(
    texts=[speaker2_text],
    base_path=sample_path,
    speaker_ids=[1],
    attention_trim=True,
    isKorean=True
)[0]
print("화자2 완료!")

print(f"\n음성 파일이 {sample_path} 폴더에 저장되었습니다.")
