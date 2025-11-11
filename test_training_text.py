# coding: utf-8
import os

from synthesizer import Synthesizer

# train.txt에 실제로 있는 문장들
training_texts = [
    "세월호 참사는 오늘로 백육십일째를 맞았습니다",  # train.txt line 30
    "오늘 뉴스룸이 주목한다 던어는 저돌입니다",  # train.txt line 1
]

load_path = "logdir-tacotron2/moon+son_2025-11-09_02-06-17"
sample_path = "logdir-tacotron2/training_text_samples"
os.makedirs(sample_path, exist_ok=True)

synthesizer = Synthesizer()
print("모델 로딩 중...")
synthesizer.load(load_path, num_speakers=2, checkpoint_step=40000, inference_prenet_dropout=False)

for i, text in enumerate(training_texts):
    print(f"\n테스트 {i+1}: {text}")

    # 화자 0으로 테스트
    print(f"  화자 0으로 합성 중...")
    audio = synthesizer.synthesize(
        texts=[text],
        base_path=sample_path,
        speaker_ids=[0],
        attention_trim=True,
        isKorean=True
    )[0]
    print(f"  완료!")

print(f"\n음성 파일이 {sample_path} 폴더에 저장되었습니다.")
print("학습 데이터에 있는 문장이므로 더 잘 합성될 것입니다.")
