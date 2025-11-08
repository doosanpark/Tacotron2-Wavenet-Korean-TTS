#  coding: utf-8
"""
TensorFlow 2.x native implementation of WaveNet operations
"""
import tensorflow as tf
import numpy as np

# Optimizer factory functions - TF2 style
def create_adam_optimizer(learning_rate, momentum):
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        epsilon=1e-4
    )


def create_sgd_optimizer(learning_rate, momentum):
    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum
    )


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        momentum=momentum,
        epsilon=1e-5
    )


optimizer_factory = {
    'adam': create_adam_optimizer,
    'sgd': create_sgd_optimizer,
    'rmsprop': create_rmsprop_optimizer
}


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes using mu-law encoding.

    Args:
        audio: Audio tensor
        quantization_channels: Number of quantization channels (typically 256)

    Returns:
        Quantized audio as int32 tensor
    '''
    mu = tf.cast(quantization_channels - 1, tf.float32)
    # Perform mu-law companding transformation (ITU-T, 1988).
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.math.log1p(mu * safe_audio_abs) / tf.math.log1p(mu)
    signal = tf.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)


def mu_law_decode(output, quantization_channels, quantization=True):
    '''Recovers waveform from mu-law quantized values.

    Args:
        output: Quantized audio tensor
        quantization_channels: Number of quantization channels
        quantization: Whether input is quantized

    Returns:
        Decoded audio tensor
    '''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    if quantization:
        signal = 2 * (tf.cast(output, tf.float32) / mu) - 1
    else:
        signal = output
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**tf.abs(signal) - 1)
    return tf.sign(signal) * magnitude


class SubPixelConvolution(tf.keras.layers.Layer):
    '''Sub-Pixel Convolutions for upsampling (TF2 Keras Layer).

    Sub-Pixel Convolutions are vanilla convolutions followed by Periodic Shuffle.
    They serve the purpose of upsampling but are faster and less prone to
    checkerboard artifacts with proper initialization.
    '''

    def __init__(self, filters, kernel_size, padding, strides,
                 NN_init=True, NN_scaler=0.3, up_layers=1, name=None, **kwargs):
        super(SubPixelConvolution, self).__init__(name=name, **kwargs)

        # Output channels = filters * H_upsample * W_upsample
        conv_filters = filters * strides[0] * strides[1]

        self.NN_init = NN_init
        self.up_layers = up_layers
        self.NN_scaler = NN_scaler
        self.out_filters = filters
        self.shuffle_strides = strides
        self.kernel_size = kernel_size

        # Create initial kernel if needed
        if NN_init:
            init_kernel = self._init_kernel(kernel_size, strides, conv_filters)
            kernel_initializer = tf.constant_initializer(init_kernel)
        else:
            kernel_initializer = 'glorot_uniform'

        # Build convolution component
        self.conv = tf.keras.layers.Conv2D(
            filters=conv_filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding=padding,
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            data_format='channels_last',
            name=f'{name}_conv' if name else 'conv'
        )

    def build(self, input_shape):
        '''Build the layer.'''
        self.conv.build(input_shape)

        if not self.NN_init:
            # If no NN init, ensure all channel-wise parameters are equal
            # Get W_0 which is the first filter of the first output channel
            W_0 = self.conv.kernel[:, :, :, 0:1]
            # Tile W_0 across all output channels
            new_kernel = tf.tile(W_0, [1, 1, 1, self.conv.filters])
            self.conv.kernel.assign(new_kernel)

        super(SubPixelConvolution, self).build(input_shape)

    def call(self, inputs, training=None):
        '''Forward pass.

        Args:
            inputs: Input tensor [batch_size, freq, time_steps, channels]

        Returns:
            Upsampled tensor [batch_size, up_freq, up_time_steps, channels]
        '''
        # Apply convolution
        convolved = self.conv(inputs, training=training)
        # Apply periodic shuffle
        return self._phase_shuffle(convolved)

    def _phase_shuffle(self, inputs):
        '''Apply periodic shuffle to upsample.'''
        # Get shapes
        batch_size = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        C = inputs.shape[-1]
        r1, r2 = self.shuffle_strides
        out_c = self.out_filters

        # Verify dimensions
        tf.debugging.assert_equal(
            C, r1 * r2 * out_c,
            message=f'Channel dimension {C} must equal {r1}*{r2}*{out_c}'
        )

        # Split and shuffle channels separately
        Xc = tf.split(inputs, out_c, axis=3)  # out_c x [batch, H, W, C/out_c]

        outputs = tf.concat([
            self._phase_shift_single(x, batch_size, H, W, r1, r2)
            for x in Xc
        ], axis=3)

        return outputs

    def _phase_shift_single(self, inputs, batch_size, H, W, r1, r2):
        '''Phase shift operation on a single channel group.'''
        # Reshape for shuffling
        x = tf.reshape(inputs, [batch_size, H, W, r1, r2])

        # Width dimension shuffle
        x = tf.transpose(x, [4, 2, 3, 1, 0])  # [r2, W, r1, H, batch]
        x = tf.batch_to_space(x, [r2], [[0, 0]])  # [1, r2*W, r1, H, batch]
        x = tf.squeeze(x, [0])  # [r2*W, r1, H, batch]

        # Height dimension shuffle
        x = tf.transpose(x, [1, 2, 0, 3])  # [r1, H, r2*W, batch]
        x = tf.batch_to_space(x, [r1], [[0, 0]])  # [1, r1*H, r2*W, batch]
        x = tf.transpose(x, [3, 1, 2, 0])  # [batch, r1*H, r2*W, 1]

        return x

    def _init_kernel(self, kernel_size, strides, filters):
        '''Initialize kernel for Nearest Neighbor upsampling (checkerboard-free).'''
        overlap = kernel_size[1] // strides[1]
        init_kernel = np.zeros(kernel_size, dtype=np.float32)

        i = kernel_size[1] // 2
        j = ([kernel_size[0] // 2 - 1, kernel_size[0] // 2]
             if kernel_size[0] % 2 == 0
             else [kernel_size[0] // 2])

        for j_i in j:
            init_kernel[j_i, i] = (1. / max(overlap, 1.)
                                   if kernel_size[1] % 2 == 0
                                   else 1.)

        # Expand to all input/output channels
        init_kernel = np.tile(
            np.expand_dims(init_kernel, 2),
            [1, 1, 1, filters]
        )

        # Apply scaling
        scale = (self.NN_scaler) ** (1 / self.up_layers)
        return init_kernel * scale

    def get_config(self):
        '''Get layer configuration for serialization.'''
        config = super(SubPixelConvolution, self).get_config()
        config.update({
            'filters': self.out_filters,
            'kernel_size': self.kernel_size,
            'strides': self.shuffle_strides,
            'NN_init': self.NN_init,
            'NN_scaler': self.NN_scaler,
            'up_layers': self.up_layers,
        })
        return config
