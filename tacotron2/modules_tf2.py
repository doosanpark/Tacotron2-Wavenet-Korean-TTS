# coding: utf-8
"""
TensorFlow 2.x native implementation of Tacotron2 modules
"""
import tensorflow as tf


class PreNet(tf.keras.layers.Layer):
    """Pre-network layer with dropout applied at both training and inference.

    This is a key component of Tacotron2 where dropout is always applied
    to introduce variation during inference.
    """

    def __init__(self, layer_sizes, drop_prob, name='prenet', **kwargs):
        super(PreNet, self).__init__(name=name, **kwargs)
        self.layer_sizes = layer_sizes
        self.drop_prob = drop_prob

        self.dense_layers = []
        for i, size in enumerate(layer_sizes):
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    units=size,
                    activation='relu',
                    name=f'dense_{i+1}'
                )
            )

        # Dropout is always applied (training=True even at inference)
        self.dropout_layers = []
        for i in range(len(layer_sizes)):
            self.dropout_layers.append(
                tf.keras.layers.Dropout(
                    rate=drop_prob,
                    name=f'dropout_{i+1}'
                )
            )

    def call(self, inputs, training=None):
        """Forward pass with dropout always enabled."""
        x = inputs
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            # Key: dropout is ALWAYS applied (even during inference)
            x = dropout(x, training=True)
        return x

    def get_config(self):
        config = super(PreNet, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'drop_prob': self.drop_prob,
        })
        return config


class HighwayNet(tf.keras.layers.Layer):
    """Highway Network layer."""

    def __init__(self, units, name='highway', **kwargs):
        super(HighwayNet, self).__init__(name=name, **kwargs)
        self.units = units

        self.H = tf.keras.layers.Dense(
            units=units,
            activation='relu',
            name='H'
        )
        self.T = tf.keras.layers.Dense(
            units=units,
            activation='sigmoid',
            bias_initializer=tf.constant_initializer(-1.0),
            name='T'
        )

    def call(self, inputs):
        """Forward pass: H * T + inputs * (1 - T)"""
        H = self.H(inputs)
        T = self.T(inputs)
        return H * T + inputs * (1.0 - T)

    def get_config(self):
        config = super(HighwayNet, self).get_config()
        config.update({'units': self.units})
        return config


class Conv1DBank(tf.keras.layers.Layer):
    """Convolution bank: K sets of 1-D convolutions."""

    def __init__(self, bank_size, num_channels, is_training, name='conv_bank', **kwargs):
        super(Conv1DBank, self).__init__(name=name, **kwargs)
        self.bank_size = bank_size
        self.num_channels = num_channels
        self.is_training = is_training

        self.conv_layers = []
        self.bn_layers = []

        for k in range(1, bank_size + 1):
            conv = tf.keras.layers.Conv1D(
                filters=num_channels,
                kernel_size=k,
                padding='same',
                activation='relu',
                name=f'conv1d_{k}'
            )
            bn = tf.keras.layers.BatchNormalization(name=f'bn_{k}')

            self.conv_layers.append(conv)
            self.bn_layers.append(bn)

    def call(self, inputs, training=None):
        """Forward pass: concatenate outputs from all conv layers."""
        if training is None:
            training = self.is_training

        outputs = []
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(inputs)
            x = bn(x, training=training)
            outputs.append(x)

        return tf.concat(outputs, axis=-1)

    def get_config(self):
        config = super(Conv1DBank, self).get_config()
        config.update({
            'bank_size': self.bank_size,
            'num_channels': self.num_channels,
            'is_training': self.is_training,
        })
        return config


class CBHG(tf.keras.layers.Layer):
    """CBHG module: Conv1D Bank + Highway + Bidirectional GRU.

    This is a key building block in Tacotron for extracting representations.
    """

    def __init__(self, bank_size, bank_channel_size, maxpool_width,
                 highway_depth, rnn_size, proj_sizes, proj_width,
                 is_training, name='cbhg', **kwargs):
        super(CBHG, self).__init__(name=name, **kwargs)

        self.bank_size = bank_size
        self.bank_channel_size = bank_channel_size
        self.maxpool_width = maxpool_width
        self.highway_depth = highway_depth
        self.rnn_size = rnn_size
        self.proj_sizes = proj_sizes
        self.proj_width = proj_width
        self.is_training = is_training

        # Convolution bank
        self.conv_bank = Conv1DBank(
            bank_size, bank_channel_size, is_training,
            name='conv_bank'
        )

        # Max pooling
        self.maxpool = tf.keras.layers.MaxPooling1D(
            pool_size=maxpool_width,
            strides=1,
            padding='same',
            name='maxpool'
        )

        # Projection layers
        self.proj_layers = []
        self.proj_bn_layers = []
        for idx, proj_size in enumerate(proj_sizes):
            activation = None if idx == len(proj_sizes) - 1 else 'relu'

            conv = tf.keras.layers.Conv1D(
                filters=proj_size,
                kernel_size=proj_width,
                padding='same',
                activation=activation,
                name=f'proj_{idx+1}'
            )
            bn = tf.keras.layers.BatchNormalization(name=f'proj_bn_{idx+1}')

            self.proj_layers.append(conv)
            self.proj_bn_layers.append(bn)

        # Highway layers
        self.highway_layers = []
        for idx in range(highway_depth):
            self.highway_layers.append(
                HighwayNet(rnn_size, name=f'highway_{idx+1}')
            )

        # Bidirectional GRU
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                rnn_size,
                return_sequences=True,
                name='gru'
            ),
            name='bi_gru'
        )

        # Optional projection to match rnn_size
        self.highway_projection = None

    def build(self, input_shape):
        """Build the layer and create projection if needed."""
        super(CBHG, self).build(input_shape)

        # Determine if we need projection before highway
        # This will be computed dynamically in call()

    def call(self, inputs, input_lengths=None, training=None):
        """Forward pass through CBHG.

        Args:
            inputs: Input tensor [batch, time, features]
            input_lengths: Sequence lengths for masking (optional)
            training: Training mode flag

        Returns:
            Output tensor [batch, time, rnn_size*2]
        """
        if training is None:
            training = self.is_training

        # Convolution bank
        conv_out = self.conv_bank(inputs, training=training)

        # Max pooling
        pool_out = self.maxpool(conv_out)

        # Projection layers
        proj_out = pool_out
        for conv, bn in zip(self.proj_layers, self.proj_bn_layers):
            proj_out = conv(proj_out)
            proj_out = bn(proj_out, training=training)

        # Residual connection
        highway_input = proj_out + inputs

        # Handle dimensionality mismatch
        if highway_input.shape[-1] != self.rnn_size:
            if self.highway_projection is None:
                self.highway_projection = tf.keras.layers.Dense(
                    self.rnn_size,
                    name='highway_projection'
                )
            highway_input = self.highway_projection(highway_input)

        # Highway layers
        highway_out = highway_input
        for highway in self.highway_layers:
            highway_out = highway(highway_out)

        # Bidirectional GRU
        if input_lengths is not None:
            # Create mask from lengths
            mask = tf.sequence_mask(
                input_lengths,
                maxlen=tf.shape(inputs)[1],
                dtype=tf.bool
            )
            outputs = self.bi_gru(highway_out, mask=mask, training=training)
        else:
            outputs = self.bi_gru(highway_out, training=training)

        return outputs

    def get_config(self):
        config = super(CBHG, self).get_config()
        config.update({
            'bank_size': self.bank_size,
            'bank_channel_size': self.bank_channel_size,
            'maxpool_width': self.maxpool_width,
            'highway_depth': self.highway_depth,
            'rnn_size': self.rnn_size,
            'proj_sizes': self.proj_sizes,
            'proj_width': self.proj_width,
            'is_training': self.is_training,
        })
        return config


# Helper function for backward compatibility
def prenet(inputs, is_training, layer_sizes, drop_prob, scope=None):
    """Functional interface for PreNet (deprecated, use PreNet layer)."""
    layer = PreNet(layer_sizes, drop_prob, name=scope or 'prenet')
    return layer(inputs, training=is_training)


def cbhg(inputs, input_lengths, is_training, bank_size, bank_channel_size,
         maxpool_width, highway_depth, rnn_size, proj_sizes, proj_width,
         scope, before_highway=None, encoder_rnn_init_state=None):
    """Functional interface for CBHG (deprecated, use CBHG layer)."""
    layer = CBHG(
        bank_size, bank_channel_size, maxpool_width, highway_depth,
        rnn_size, proj_sizes, proj_width, is_training, name=scope
    )
    return layer(inputs, input_lengths, training=is_training)
