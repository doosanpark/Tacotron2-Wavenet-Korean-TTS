# coding: utf-8
"""
Guided Attention Loss for Tacotron2
논문: "Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention"
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def guided_attention(attention_weights, input_lengths, target_lengths, g=0.2):
    """
    Guided attention loss를 계산합니다.

    Args:
        attention_weights: [batch_size, decoder_steps, encoder_steps] attention 가중치
        input_lengths: [batch_size] 입력 시퀀스 길이
        target_lengths: [batch_size] 출력 시퀀스 길이
        g: float, guided attention의 강도 (기본값: 0.2)

    Returns:
        loss: scalar, guided attention loss
    """
    batch_size = tf.shape(attention_weights)[0]
    max_decoder_steps = tf.shape(attention_weights)[1]
    max_encoder_steps = tf.shape(attention_weights)[2]

    # N: encoder steps (text length)
    # T: decoder steps (audio length)
    N = tf.cast(max_encoder_steps, tf.float32)
    T = tf.cast(max_decoder_steps, tf.float32)

    # n, t 그리드 생성
    n = tf.range(max_encoder_steps, dtype=tf.float32)  # [0, 1, 2, ..., N-1]
    t = tf.range(max_decoder_steps, dtype=tf.float32)  # [0, 1, 2, ..., T-1]

    # W[t, n] = 1 - exp(-(n/N - t/T)^2 / (2*g^2))
    # 대각선에 가까울수록 0에 가까운 값, 멀어질수록 1에 가까운 값
    n_normalized = n / (N + 1e-6)  # [max_encoder_steps]
    t_normalized = t / (T + 1e-6)  # [max_decoder_steps]

    # Broadcasting: [max_decoder_steps, 1] - [1, max_encoder_steps] = [max_decoder_steps, max_encoder_steps]
    n_normalized = tf.reshape(n_normalized, [1, 1, max_encoder_steps])
    t_normalized = tf.reshape(t_normalized, [1, max_decoder_steps, 1])

    W = 1.0 - tf.exp(-tf.square(n_normalized - t_normalized) / (2.0 * g * g))

    # Masking: 실제 길이를 넘어서는 부분은 무시
    # input mask: [batch_size, 1, max_encoder_steps]
    input_mask = tf.sequence_mask(
        input_lengths, maxlen=max_encoder_steps, dtype=tf.float32)
    input_mask = tf.expand_dims(input_mask, 1)

    # target mask: [batch_size, max_decoder_steps, 1]
    target_mask = tf.sequence_mask(
        target_lengths, maxlen=max_decoder_steps, dtype=tf.float32)
    target_mask = tf.expand_dims(target_mask, 2)

    # Combined mask: [batch_size, max_decoder_steps, max_encoder_steps]
    mask = input_mask * target_mask

    # Guided attention loss
    # attention_weights: [batch_size, decoder_steps, encoder_steps]
    # W: [1, decoder_steps, encoder_steps]
    # mask: [batch_size, decoder_steps, encoder_steps]
    loss = attention_weights * W * mask

    # Average over valid positions
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    return loss


if __name__ == "__main__":
    # 테스트
    print("Guided Attention Loss 테스트")

    # 더미 데이터
    batch_size = 2
    max_decoder_steps = 100
    max_encoder_steps = 20

    # 랜덤 attention weights (정상적으로 정렬되지 않은 경우)
    attention_weights = tf.random.normal([batch_size, max_decoder_steps, max_encoder_steps])
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)

    input_lengths = tf.constant([20, 15])
    target_lengths = tf.constant([100, 80])

    loss = guided_attention(attention_weights, input_lengths, target_lengths)

    with tf.Session() as sess:
        loss_value = sess.run(loss)
        print(f"Guided Attention Loss: {loss_value}")
        print("\n이 loss를 학습 중 total loss에 추가하면 attention 정렬이 개선됩니다.")
