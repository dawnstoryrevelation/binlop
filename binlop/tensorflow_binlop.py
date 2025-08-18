"""TensorFlow implementation of the BiNLOP activation.

This module defines a Keras layer implementing the BiNLOP activation. It
closely follows the PyTorch implementation but uses TensorFlow primitives
for computation. The layer exposes the same operating modes:
``standard_certified`` (default), ``flow_safe`` and ``standard_unconstrained``.

The certified mode applies a per‑batch clamp to the gating parameters to
ensure the derivative remains within strict bounds. The flow safe mode
uses constant mixture weights and yields an exact diagonal Jacobian. The
unconstrained mode uses adaptive gating without certification.

Example
-------

>>> import tensorflow as tf
>>> from binlop.tensorflow_binlop import Binlop
>>> act = Binlop(mode='standard_certified')
>>> x = tf.random.normal([32, 64])
>>> y = act(x)

This layer is compatible with the Keras functional and subclassing APIs. It
supports mixed precision by computing intermediate results in float32 and
casting the output back to the input dtype.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import tensorflow as tf


def _tf_phi(x: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Evaluate the rational basis functions φ_i for each component.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of arbitrary shape.
    a : tf.Tensor
        Tensor of shape ``(K,)`` containing the ``a_i`` coefficients.
    b : tf.Tensor
        Tensor of shape ``(K,)`` containing the ``b_i`` coefficients.

    Returns
    -------
    tf.Tensor
        A tensor of shape ``(K,) + x.shape`` containing φ_i(x) for each
        component along the first dimension.
    """
    x2 = tf.square(x)
    # Reshape a and b to broadcast over x
    a_broadcast = tf.reshape(a, (tf.shape(a)[0],) + (1,) * tf.rank(x))
    b_broadcast = tf.reshape(b, (tf.shape(b)[0],) + (1,) * tf.rank(x))
    x_expand = tf.expand_dims(x, axis=0)
    x2_expand = tf.expand_dims(x2, axis=0)
    num = 1.0 + a_broadcast * x2_expand
    denom = 1.0 + b_broadcast * x2_expand
    return x_expand * (num / denom)


def _tf_phi_prime(x: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Compute the derivative φ_i'(x) for each component.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of arbitrary shape.
    a : tf.Tensor
        Tensor of shape ``(K,)`` containing the ``a_i`` coefficients.
    b : tf.Tensor
        Tensor of shape ``(K,)`` containing the ``b_i`` coefficients.

    Returns
    -------
    tf.Tensor
        A tensor of shape ``(K,) + x.shape`` containing φ_i'(x) for each
        component along the first dimension.
    """
    x2 = tf.square(x)
    a_broadcast = tf.reshape(a, (tf.shape(a)[0],) + (1,) * tf.rank(x))
    b_broadcast = tf.reshape(b, (tf.shape(b)[0],) + (1,) * tf.rank(x))
    x2_expand = tf.expand_dims(x2, axis=0)
    denom = 1.0 + b_broadcast * x2_expand
    num = 1.0 + (3.0 * a_broadcast - b_broadcast) * x2_expand + (a_broadcast * b_broadcast) * tf.square(x2_expand)
    return num / tf.square(denom)


class Binlop(tf.keras.layers.Layer):
    """BiNLOP activation layer for TensorFlow.

    Parameters
    ----------
    gamma_min : float, optional
        Minimum shared tail slope. Must be > 1/9. Defaults to 0.6.
    gamma_init : float, optional
        Initial shared tail slope for the learnable parameter ``g_hat``.
        Defaults to 0.8.
    k_inits : Iterable[float], optional
        Initial knee positions for each component. Defines the number of
        mixture components ``K``. Defaults to (1.5, 4.0).
    mode : {"standard_certified", "flow_safe", "standard_unconstrained"}, optional
        Operating mode for the activation. Defaults to "standard_certified".
    eps : float, optional
        Numerical epsilon used for stability. Defaults to 1e-6.

    Notes
    -----
    Unlike the PyTorch implementation, this layer does not provide an
    ``invert`` method. Inversion is rarely required in TensorFlow models
    because reversible networks typically implement custom inverses.
    """

    def __init__(self,
                 gamma_min: float = 0.6,
                 gamma_init: float = 0.8,
                 k_inits: Iterable[float] = (1.5, 4.0),
                 mode: str = "standard_certified",
                 eps: float = 1e-6,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if not gamma_min > (1.0 / 9.0) + 1e-6:
            raise ValueError("gamma_min must be > 1/9 to ensure strict monotonicity")
        if not (0.0 < gamma_init <= 1.0):
            raise ValueError("gamma_init must lie in (0, 1]")
        if mode not in {"standard_certified", "flow_safe", "standard_unconstrained"}:
            raise ValueError(f"Invalid mode '{mode}'")
        self.gamma_min = float(gamma_min)
        self.gamma_init = float(gamma_init)
        self.k_inits = tuple(float(k) for k in k_inits)
        self.mode = mode
        self.eps = float(eps)
        self.K = len(self.k_inits)

    def build(self, input_shape):
        # Parameterise gamma via g_hat
        p = (self.gamma_init - self.gamma_min) / (1.0 - self.gamma_min)
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        self.g_hat = self.add_weight(
            name="g_hat",
            shape=(),
            initializer=tf.keras.initializers.Constant(tf.math.log(p / (1 - p))),
            trainable=True,
        )
        # Knees parameterised in log space
        s0 = [math.log(k) for k in self.k_inits]
        self.s_hat = self.add_weight(
            name="s_hat",
            shape=(self.K,),
            initializer=tf.keras.initializers.Constant(s0),
            trainable=True,
        )
        # Gating parameters
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.K,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.K,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        # Flow safe weights
        if self.mode == "flow_safe":
            self.w_logits = self.add_weight(
                name="w_logits",
                shape=(self.K,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )
        super().build(input_shape)

    def _params(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        gamma = self.gamma_min + (1.0 - self.gamma_min) * tf.math.sigmoid(self.g_hat)
        k = tf.exp(self.s_hat) + self.eps
        b = 3.0 / (tf.square(k) + self.eps)
        a = gamma * b
        return gamma, a, b

    def _context(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x32 = tf.cast(x, tf.float32)
        mean_x2 = tf.reduce_mean(tf.square(x32))
        c = tf.math.log(mean_x2 + self.eps)
        return c, mean_x2

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.cast(inputs, tf.float32)
        gamma, a, b = self._params()

        if self.mode == "flow_safe":
            w = tf.nn.softmax(self.w_logits)
        else:
            c, _ = self._context(x)
            logits = self.alpha * c + self.beta
            w_hat = tf.nn.softmax(logits)
            if self.mode == "standard_unconstrained":
                w = w_hat
            else:
                # Certified mode: compute scale factor s
                # Compute derivative mixture
                phi_prime_all = _tf_phi_prime(x, a, b)  # shape (K, ...) 
                # Broadcast weights
                w_broadcast = tf.reshape(w_hat, (self.K,) + (1,) * tf.rank(x))
                phi_prime_mix = tf.reduce_sum(w_broadcast * phi_prime_all, axis=0)
                margin = 1.0 - phi_prime_mix
                min_margin = tf.reduce_min(margin)
                min_pos = tf.reduce_min(phi_prime_mix)
                # Sensitivity of weights
                alpha_mean = tf.reduce_sum(w_hat * self.alpha)
                dw_dc = w_hat * (self.alpha - alpha_mean)
                L_w = tf.reduce_sum(tf.abs(dw_dc)) + 1e-12
                # Mean and max squared magnitude
                x2 = tf.square(x)
                mean_x2 = tf.reduce_mean(x2)
                max_x2 = tf.reduce_max(x2)
                M = tf.cast(tf.size(x), tf.float32)
                delta_bound = L_w * 2.0 * max_x2 / (M * (mean_x2 + self.eps))
                m = (9.0 * gamma - 1.0) / 8.0
                tau = 0.05 * m
                s1 = min_margin / (delta_bound + 1e-12)
                s2 = (min_pos - tau) / (delta_bound + 1e-12)
                # s = min(max(0, min(s1, s2)), 1)
                s = tf.minimum(1.0, tf.maximum(0.0, tf.minimum(s1, s2)))
                # Rescale alpha and recompute weights if s < 1
                w = tf.cond(
                    tf.less(s, 1.0),
                    lambda: tf.nn.softmax((self.alpha * s) * c + self.beta),
                    lambda: w_hat,
                )
        # Compute φ_i(x)
        phi_components = _tf_phi(x, a, b)  # shape (K, ...) 
        w_broadcast = tf.reshape(w, (self.K,) + (1,) * tf.rank(x))
        y = tf.reduce_sum(w_broadcast * phi_components, axis=0)
        return tf.cast(y, inputs.dtype)
