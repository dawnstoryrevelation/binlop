import tensorflow as tf

@tf.custom_gradient
def binlop_op(x, gamma, k):
    # Forward operation
    y = gamma * x + (1.0 - gamma) * tf.clip_by_value(x, -k, k)

    def grad(dy):
        # Gradient backward similar to PyTorch implementation
        M = tf.cast(tf.abs(y) > k, dy.dtype)
        signy = tf.sign(y)

        dx = dy * (1.0 - (1.0 - gamma) * M)
        clamp_y = tf.clip_by_value(y, -k, k)
        dgamma = dy * ((y - clamp_y) / (gamma + 1e-12)) * M
        dk = dy * (1.0 - gamma) * signy * M

        # Sum gradients over broadcasted dimensions for gamma and k
        # Reduce dims if needed according to the shapes of gamma and k
        while dgamma.shape.ndims > gamma.shape.ndims:
            dgamma = tf.reduce_sum(dgamma, axis=0)
            dk = tf.reduce_sum(dk, axis=0)

        # No explicit sum across parameter dims if not needed, relying on broadcasting

        return dx, dgamma, dk

    return y, grad


class BiNLOP2(tf.Module):
    """
    BiNLOP-2: φ(x) = γ x + (1 − γ) clamp(x, −k, k)
    Odd, monotone, bi-Lipschitz, invertible; dtype-preserving; memoryless backward.
    """
    def __init__(self, gamma_min=0.6, gamma_init=0.85, k_init=2.0, per_channel=False, num_channels=None, name=None):
        super().__init__(name=name)
        assert 0.0 < gamma_min < 1.0 and gamma_min < gamma_init < 1.0
        self.gamma_min = gamma_min
        self.per_channel = per_channel

        def clip_p(p):
            return max(min(p, 1 - 1e-6), 1e-6)

        p = (gamma_init - gamma_min) / (1 - gamma_min)
        p = clip_p(p)

        if per_channel:
            assert num_channels and num_channels > 0
            init_g = tf.fill([num_channels], tf.math.log(p / (1 - p)))  # inverse sigmoid / logit
            init_s = tf.fill([num_channels], tf.math.log(k_init))

            self.g_hat = tf.Variable(init_g, dtype=tf.float32, trainable=True)
            self.s_hat = tf.Variable(init_s, dtype=tf.float32, trainable=True)
        else:
            self.g_hat = tf.Variable(tf.math.log(p / (1 - p)), dtype=tf.float32, trainable=True)
            self.s_hat = tf.Variable(tf.math.log(k_init), dtype=tf.float32, trainable=True)

    def _params(self, x):
        gamma = self.gamma_min + (1.0 - self.gamma_min) * tf.sigmoid(self.g_hat)
        k = tf.exp(self.s_hat)

        if self.per_channel:
            # Reshape gamma and k to be broadcastable with input x
            if len(x.shape) == 4:   # NCHW or NHWC? TensorFlow default is NHWC
                # Assuming NHWC: shape = (batch, height, width, channels)
                gamma = tf.reshape(gamma, [1, 1, 1, -1])
                k = tf.reshape(k, [1, 1, 1, -1])
            elif len(x.shape) == 3: # NLC or NCL shape is less common in TF, guessing NLC (batch, length, channels)
                gamma = tf.reshape(gamma, [1, 1, -1])
                k = tf.reshape(k, [1, 1, -1])
        return tf.cast(gamma, x.dtype), tf.cast(k, x.dtype)

    def __call__(self, x):
        gamma, k = self._params(x)
        return binlop_op(x, gamma, k)

    @tf.function
    def invert(self, y):
        gamma, k = self._params(y)
        s = tf.abs(y)
        core = s <= k
        inv_gamma = 1.0 / gamma
        x = tf.where(core, y, tf.sign(y) * (k + (s - k) * inv_gamma))
        return x

    @tf.function
    def logdet(self, x):
        gamma, k = self._params(x)
        mask = tf.cast(tf.abs(x) > k, x.dtype)
        return tf.reduce_sum(mask * tf.math.log(gamma))
