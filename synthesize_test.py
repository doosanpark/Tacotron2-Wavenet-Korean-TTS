# coding: utf-8
import os
import sys

# 간단한 텍스트로 테스트
test_text = "안녕하세요"

print(f"테스트 텍스트: {test_text}")

from synthesizer import Synthesizer

# 체크포인트 20000으로 시도
load_path = "logdir-tacotron2/moon+son_2025-11-09_02-06-17"
sample_path = "logdir-tacotron2/test_samples"
os.makedirs(sample_path, exist_ok=True)

synthesizer = Synthesizer()
print("모델 로딩 중 (checkpoint 20000)...")
synthesizer.load(load_path, num_speakers=2, checkpoint_step=20000, inference_prenet_dropout=False)

# 화자 0으로 테스트
print("\n음성 합성 중...")
audio = synthesizer.synthesize(
    texts=[test_text],
    base_path=sample_path,
    speaker_ids=[0],
    attention_trim=True,
    isKorean=True
)[0]
print("완료!")

print(f"\n음성 파일이 {sample_path} 폴더에 저장되었습니다.")
