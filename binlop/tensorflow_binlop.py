import tensorflow as tf
from tensorflow.keras import layers

class BiNLOP(tf.keras.layers.Layer):
    """
    BiNLOP-1: Linear-Tail Hinge
    phi(x) = x - (1 - gamma) * sign(x) * relu(|x| - k)
    Guarantees: odd, monotone, bi-Lipschitz with phi' in {1, gamma}, invertible.
    """

    def __init__(self, gamma_min=0.6, gamma_init=0.8, k_init=2.0,
                 per_channel=False, num_channels=None, **kwargs):
        super(BiNLOP, self).__init__(**kwargs)

        assert 0.0 < gamma_min < 1.0 and gamma_min < gamma_init < 1.0
        self.gamma_min = float(gamma_min)
        self.per_channel = per_channel
        self.num_channels = num_channels

        # Convert gamma_init to probability form (same as PyTorch version)
        p = (gamma_init - gamma_min) / (1.0 - gamma_min)
        p = float(max(min(p, 1 - 1e-6), 1e-6))
        g_hat_init = tf.math.log(p) - tf.math.log(1.0 - p)  # logit
        s_hat_init = tf.math.log(k_init)

        if per_channel:
            assert num_channels is not None and num_channels > 0
            g_shape = (num_channels,)
            s_shape = (num_channels,)
        else:
            g_shape = ()
            s_shape = ()

        self.g_hat = tf.Variable(initial_value=tf.fill(g_shape, g_hat_init),
                                 trainable=True, dtype=tf.float32, name="g_hat")
        self.s_hat = tf.Variable(initial_value=tf.fill(s_shape, s_hat_init),
                                 trainable=True, dtype=tf.float32, name="s_hat")

    def _params(self, x):
        gamma = self.gamma_min + (1.0 - self.gamma_min) * tf.sigmoid(self.g_hat)
        k = tf.exp(self.s_hat)

        if self.per_channel:
            if len(x.shape) == 4:  # NHWC format in TF (not NCHW!)
                gamma = tf.reshape(gamma, (1, 1, 1, -1))
                k = tf.reshape(k, (1, 1, 1, -1))
            elif len(x.shape) == 3:  # NLC
                gamma = tf.reshape(gamma, (1, 1, -1))
                k = tf.reshape(k, (1, 1, -1))

        return gamma, k

    def call(self, x):
        x32 = tf.cast(x, tf.float32)  # internal computations in float32
        gamma, k = self._params(x32)
        s = tf.abs(x32)
        t = tf.nn.relu(s - k)
        y = x32 - (1.0 - gamma) * tf.sign(x32) * t
        return tf.cast(y, x.dtype)

    def invert(self, y):
        y32 = tf.cast(y, tf.float32)
        gamma, k = self._params(y32)
        s = tf.abs(y32)
        core = s <= k
        inv_gamma = 1.0 / gamma
        x_core = y32
        x_tail = tf.sign(y32) * (k + (s - k) * inv_gamma)
        x = tf.where(core, x_core, x_tail)
        return tf.cast(x, y.dtype)
