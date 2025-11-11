# coding: utf-8
import sys
import numpy as np
from text import text_to_sequence, sequence_to_text

# 테스트 텍스트
test_texts = [
    "안녕하세요. 저는 박두산입니다.",
    "만나서 반갑습니다. 저는 산두박입니다."
]

print("=== 텍스트 변환 디버깅 ===\n")

for i, text in enumerate(test_texts, 1):
    print(f"화자{i} 원본 텍스트: {text}")

    # 텍스트를 시퀀스로 변환
    sequence = text_to_sequence(text)
    print(f"변환된 시퀀스 길이: {len(sequence)}")
    print(f"시퀀스 내용: {sequence[:50]}...")  # 처음 50개만 출력

    # 시퀀스를 다시 텍스트로 변환
    recovered_text = sequence_to_text(sequence)
    print(f"복원된 텍스트: {recovered_text}")
    print()
