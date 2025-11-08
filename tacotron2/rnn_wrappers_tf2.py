# coding: utf-8
"""
TensorFlow 2.x native implementation of RNN cells and attention mechanisms
"""
import tensorflow as tf
from tacotron2.modules_tf2 import PreNet


class ZoneoutLSTMCell(tf.keras.layers.Layer):
    """LSTM Cell with Zoneout regularization.

    Zoneout randomly preserves hidden activations instead of dropping them.
    Reference: https://arxiv.org/abs/1606.01305
    """

    def __init__(self, units, zoneout_factor_cell=0., zoneout_factor_output=0.,
                 name='zoneout_lstm', **kwargs):
        super(ZoneoutLSTMCell, self).__init__(name=name, **kwargs)

        self.units = units
        self.zoneout_cell = zoneout_factor_cell
        self.zoneout_output = zoneout_factor_output

        # Validate zoneout factors
        if min(zoneout_factor_cell, zoneout_factor_output) < 0:
            raise ValueError('Zoneout factors must be >= 0')
        if max(zoneout_factor_cell, zoneout_factor_output) > 1:
            raise ValueError('Zoneout factors must be <= 1')

        # Create underlying LSTM cell
        self.lstm_cell = tf.keras.layers.LSTMCell(units, name='lstm')

    @property
    def state_size(self):
        return self.lstm_cell.state_size

    @property
    def output_size(self):
        return self.lstm_cell.output_size

    def build(self, input_shape):
        self.lstm_cell.build(input_shape)
        super(ZoneoutLSTMCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        """Apply LSTM with zoneout.

        Args:
            inputs: Input tensor
            states: Tuple of (c_state, h_state)
            training: Training mode flag

        Returns:
            (output, new_states) tuple
        """
        # Run vanilla LSTM
        output, new_states = self.lstm_cell(inputs, states, training=training)

        # Unpack states
        prev_c, prev_h = states
        new_c, new_h = new_states

        # Apply zoneout
        if training:
            # During training, randomly preserve activations
            # Note: dropout preserves some values, zoneout preserves old values

            # Zoneout for cell state
            c_diff = new_c - prev_c
            c_mask = tf.nn.dropout(
                tf.ones_like(c_diff),
                rate=self.zoneout_cell
            )
            c = prev_c + c_diff * c_mask / (1 - self.zoneout_cell)

            # Zoneout for hidden state
            h_diff = new_h - prev_h
            h_mask = tf.nn.dropout(
                tf.ones_like(h_diff),
                rate=self.zoneout_output
            )
            h = prev_h + h_diff * h_mask / (1 - self.zoneout_output)

        else:
            # During inference, use expected value
            c = (1 - self.zoneout_cell) * new_c + self.zoneout_cell * prev_c
            h = (1 - self.zoneout_output) * new_h + self.zoneout_output * prev_h

        new_state = [c, h]
        return output, new_state

    def get_config(self):
        config = super(ZoneoutLSTMCell, self).get_config()
        config.update({
            'units': self.units,
            'zoneout_factor_cell': self.zoneout_cell,
            'zoneout_factor_output': self.zoneout_output,
        })
        return config


class LocationSensitiveAttention(tf.keras.layers.Layer):
    """Location-Sensitive Attention mechanism.

    Combines content-based (Bahdanau) and location-based attention.
    Reference: https://arxiv.org/abs/1506.07503
    """

    def __init__(self, units, memory, memory_sequence_length=None,
                 attention_filters=32, attention_kernel=31,
                 smoothing=False, cumulative_weights=True,
                 name='location_attention', **kwargs):
        super(LocationSensitiveAttention, self).__init__(name=name, **kwargs)

        self.units = units
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.cumulative_weights = cumulative_weights
        self.smoothing = smoothing

        # Query layer: projects decoder state
        self.query_layer = tf.keras.layers.Dense(
            units, use_bias=False, name='query_layer'
        )

        # Keys layer: projects memory (encoder outputs)
        self.keys_layer = tf.keras.layers.Dense(
            units, use_bias=False, name='keys_layer'
        )

        # Location convolution: processes previous attention weights
        self.location_conv = tf.keras.layers.Conv1D(
            filters=attention_filters,
            kernel_size=attention_kernel,
            padding='same',
            use_bias=True,
            bias_initializer='zeros',
            name='location_conv'
        )

        # Location projection: projects location features
        self.location_layer = tf.keras.layers.Dense(
            units, use_bias=False, name='location_layer'
        )

        # Scoring layer
        self.v_a = None  # Created in build()
        self.b_a = None  # Created in build()

    def build(self, input_shape):
        # Create scoring variables
        self.v_a = self.add_weight(
            'attention_variable',
            shape=[self.units],
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        self.b_a = self.add_weight(
            'attention_bias',
            shape=[self.units],
            initializer='zeros',
            trainable=True
        )

        # Preprocess memory (encoder outputs)
        self.keys = self.keys_layer(self.memory)

        # Create memory mask if needed
        if self.memory_sequence_length is not None:
            self.memory_mask = tf.sequence_mask(
                self.memory_sequence_length,
                maxlen=tf.shape(self.memory)[1],
                dtype=tf.bool
            )
        else:
            self.memory_mask = None

        super(LocationSensitiveAttention, self).build(input_shape)

    def call(self, query, previous_alignments, training=None):
        """Compute attention weights.

        Args:
            query: Decoder state [batch, query_depth]
            previous_alignments: Previous attention weights [batch, max_time]

        Returns:
            alignments: Current attention weights [batch, max_time]
            next_state: Cumulative or current alignments for next step
        """
        # Process query: [batch, query_depth] -> [batch, 1, units]
        processed_query = self.query_layer(query)
        processed_query = tf.expand_dims(processed_query, axis=1)

        # Process previous alignments for location features
        # [batch, max_time] -> [batch, max_time, 1]
        expanded_alignments = tf.expand_dims(previous_alignments, axis=-1)

        # Apply location convolution
        # [batch, max_time, 1] -> [batch, max_time, filters]
        location_features = self.location_conv(expanded_alignments)

        # Project location features
        # [batch, max_time, filters] -> [batch, max_time, units]
        processed_location = self.location_layer(location_features)

        # Compute energy (scores)
        # energy = v_a^T tanh(W_keys(memory) + W_query(query) + W_location(f) + b_a)
        energy = tf.reduce_sum(
            self.v_a * tf.tanh(
                self.keys + processed_query + processed_location + self.b_a
            ),
            axis=-1
        )

        # Apply mask if provided
        if self.memory_mask is not None:
            # Set masked positions to large negative value
            energy = tf.where(
                self.memory_mask,
                energy,
                tf.ones_like(energy) * (-1e9)
            )

        # Normalize to get alignments (attention weights)
        if self.smoothing:
            # Sigmoid normalization (allows attending to multiple positions)
            alignments = tf.nn.sigmoid(energy)
            alignments = alignments / tf.reduce_sum(
                alignments, axis=-1, keepdims=True
            )
        else:
            # Softmax normalization (standard)
            alignments = tf.nn.softmax(energy, axis=-1)

        # Compute next state (cumulative or current alignments)
        if self.cumulative_weights:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments

        return alignments, next_state

    def initial_alignments(self, batch_size, dtype=tf.float32):
        """Get initial alignment (all zeros)."""
        max_time = tf.shape(self.memory)[1]
        return tf.zeros([batch_size, max_time], dtype=dtype)

    def get_config(self):
        config = super(LocationSensitiveAttention, self).get_config()
        config.update({
            'units': self.units,
            'attention_filters': self.attention_filters,
            'attention_kernel': self.attention_kernel,
            'smoothing': self.smoothing,
            'cumulative_weights': self.cumulative_weights,
        })
        return config


class GMMAttention(tf.keras.layers.Layer):
    """GMM (Gaussian Mixture Model) Attention mechanism.

    Uses a mixture of Gaussians to compute attention weights.
    """

    def __init__(self, num_mixtures, memory, memory_sequence_length=None,
                 name='gmm_attention', **kwargs):
        super(GMMAttention, self).__init__(name=name, **kwargs)

        self.num_mixtures = num_mixtures
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length

        # Query projection: outputs alpha, beta, kappa parameters
        self.query_layer = tf.keras.layers.Dense(
            3 * num_mixtures,
            use_bias=True,
            name='gmm_projection'
        )

        # Precompute memory positions
        self.alignments_size = tf.shape(memory)[1]

    def build(self, input_shape):
        super(GMMAttention, self).build(input_shape)

    def call(self, query, previous_kappa, training=None):
        """Compute GMM attention weights.

        Args:
            query: Decoder state [batch, query_depth]
            previous_kappa: Previous kappa values [batch, num_mixtures]

        Returns:
            alignments: Attention weights [batch, max_time]
            next_kappa: Updated kappa values
        """
        # Project query to GMM parameters
        params = self.query_layer(query)
        alpha_hat, beta_hat, kappa_hat = tf.split(params, 3, axis=-1)

        # Transform parameters
        # Alpha: mixture weights (use exp for positive values)
        alpha = tf.expand_dims(tf.exp(alpha_hat), axis=-1)

        # Beta: inverse variance (use exp for positive values)
        beta = tf.expand_dims(tf.exp(beta_hat), axis=-1)

        # Kappa: means (cumulative, use exp for positive delta)
        kappa = tf.expand_dims(previous_kappa + tf.exp(kappa_hat), axis=-1)

        # Memory positions: [1, 1, max_time]
        mu = tf.reshape(
            tf.cast(tf.range(self.alignments_size), tf.float32),
            [1, 1, -1]
        )

        # Compute Gaussian probabilities
        # phi = sum_k alpha_k * exp(-beta_k * (kappa_k - mu)^2)
        phi = tf.reduce_sum(
            alpha * tf.exp(-beta * tf.square(kappa - mu)),
            axis=1
        )

        # Apply mask if provided
        if self.memory_sequence_length is not None:
            mask = tf.sequence_mask(
                self.memory_sequence_length,
                maxlen=self.alignments_size,
                dtype=tf.bool
            )
            phi = tf.where(mask, phi, tf.zeros_like(phi))

        # Normalize
        alignments = phi / (tf.reduce_sum(phi, axis=-1, keepdims=True) + 1e-8)

        # Update kappa
        next_kappa = tf.squeeze(kappa, axis=-1)

        return alignments, next_kappa

    def initial_state(self, batch_size, dtype=tf.float32):
        """Get initial kappa (all zeros)."""
        return tf.zeros([batch_size, self.num_mixtures], dtype=dtype)

    def get_config(self):
        config = super(GMMAttention, self).get_config()
        config.update({
            'num_mixtures': self.num_mixtures,
        })
        return config


class DecoderPrenetWrapper(tf.keras.layers.Layer):
    """Wraps a decoder cell to apply PreNet to inputs before processing.

    This is used in Tacotron2 decoder to process mel frames before
    feeding them to the LSTM.
    """

    def __init__(self, cell, prenet_sizes, dropout_prob,
                 inference_prenet_dropout=True, name='decoder_prenet', **kwargs):
        super(DecoderPrenetWrapper, self).__init__(name=name, **kwargs)

        self.cell = cell
        self.prenet_sizes = prenet_sizes
        self.dropout_prob = dropout_prob
        self.inference_prenet_dropout = inference_prenet_dropout

        # Create PreNet layer
        self.prenet = PreNet(prenet_sizes, dropout_prob, name='prenet')

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        # Output size includes both cell output and attention context
        # This will be determined at runtime
        return self.cell.output_size

    def build(self, input_shape):
        self.prenet.build(input_shape)
        super(DecoderPrenetWrapper, self).build(input_shape)

    def call(self, inputs, states, training=None):
        """Apply prenet then cell.

        Args:
            inputs: Input tensor (mel frame)
            states: Cell states
            training: Training mode flag

        Returns:
            (output, new_states) tuple
        """
        # Apply PreNet to inputs
        prenet_out = self.prenet(inputs, training=training)

        # Apply cell
        output, new_states = self.cell(prenet_out, states, training=training)

        return output, new_states

    def get_config(self):
        config = super(DecoderPrenetWrapper, self).get_config()
        config.update({
            'prenet_sizes': self.prenet_sizes,
            'dropout_prob': self.dropout_prob,
            'inference_prenet_dropout': self.inference_prenet_dropout,
        })
        return config
