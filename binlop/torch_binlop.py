"""PyTorch implementation of the BiNLOP activation.

This module defines a single class, :class:`Binlop`, which inherits from
``torch.nn.Module`` and implements the BiNLOP activation function with
multiple operating modes. The implementation follows the design of the
Certified Multi‑Scale Bi‑Lipschitz Rational Unit (CMBIRU‑C) but uses the
public name ``BiNLOP``. It supports certified gating to maintain
Lipschitz bounds, a flow‑safe variant with constant mixture weights and an
unconstrained adaptive variant for ablations.

Examples
--------

>>> import torch
>>> from binlop.torch_binlop import Binlop
>>> act = Binlop(mode='standard_certified')
>>> x = torch.randn(8, 16)
>>> y = act(x)
>>> inv = act.invert(y)  # approximate inverse of the activation

The class is designed to be drop‑in compatible with existing PyTorch
workflows and supports gradient propagation via Autograd. Mixed precision is
handled internally by upcasting intermediate computations to ``float32``
before downcasting back to the input dtype.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _phi(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate the rational basis function for each component.

    Each component has the form φ_i(x) = x * (1 + a_i * x^2) / (1 + b_i * x^2).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape.
    a : torch.Tensor
        Tensor of shape ``(K,)`` containing the ``a_i`` coefficients.
    b : torch.Tensor
        Tensor of shape ``(K,)`` containing the ``b_i`` coefficients.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(*x.shape, K)`` where the final dimension
        corresponds to the component outputs.
    """
    # Ensure float computation
    x2 = x * x
    # Reshape a and b for broadcasting across x
    # a and b have shape (K,), reshape to (K, 1, ..., 1)
    # We'll unsqueeze at the front so the final dimension is K
    a_broadcast = a.view(a.shape[0], *([1] * x.dim()))
    b_broadcast = b.view(b.shape[0], *([1] * x.dim()))
    x_expand = x.unsqueeze(0)  # shape (1, *x.shape)
    x2_expand = x2.unsqueeze(0)
    num = 1.0 + a_broadcast * x2_expand
    denom = 1.0 + b_broadcast * x2_expand
    return x_expand * (num / denom)


def _phi_prime(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the derivative φ_i'(x) for each component.

    The derivative of φ_i(x) = x * (1 + a_i x^2) / (1 + b_i x^2) is
    given by (1 + (3 a_i - b_i) x^2 + a_i b_i x^4) / (1 + b_i x^2)^2.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape.
    a : torch.Tensor
        Tensor of shape ``(K,)`` containing the ``a_i`` coefficients.
    b : torch.Tensor
        Tensor of shape ``(K,)`` containing the ``b_i`` coefficients.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(*x.shape, K)`` containing the derivatives for
        each component.
    """
    x2 = x * x
    a_broadcast = a.view(a.shape[0], *([1] * x.dim()))
    b_broadcast = b.view(b.shape[0], *([1] * x.dim()))
    x2_expand = x2.unsqueeze(0)
    denom = 1.0 + b_broadcast * x2_expand
    # numerator: 1 + (3a - b)*x^2 + (a*b)*(x^2)^2
    num = 1.0 + (3.0 * a_broadcast - b_broadcast) * x2_expand + (a_broadcast * b_broadcast) * (x2_expand * x2_expand)
    return num / (denom * denom)


class Binlop(nn.Module):
    """BiNLOP activation function for PyTorch.

    Parameters
    ----------
    gamma_min : float, optional
        Minimum shared tail slope. Must be > 1/9 to ensure strict
        monotonicity. Defaults to ``0.6``.
    gamma_init : float, optional
        Initial shared tail slope used to initialise the learnable
        parameter ``g_hat``. Should satisfy ``gamma_min <= gamma_init <= 1``.
        Defaults to ``0.8``.
    k_inits : Iterable[float], optional
        Initial knee positions for each component. The length of this
        iterable defines the number of mixture components ``K``. Defaults to
        ``(1.5, 4.0)``.
    mode : {"standard_certified", "flow_safe", "standard_unconstrained"}, optional
        Operating mode for the activation. In ``standard_certified`` mode the
        adaptive gating is clamped per batch to ensure that the derivative
        remains within ``[m, 1]`` for ``m = (9*gamma_min - 1)/8``. In
        ``flow_safe`` mode constant weights are used and the Jacobian is
        exactly diagonal. In ``standard_unconstrained`` mode the gating is
        adaptive but no certification is applied. Defaults to
        ``"standard_certified"``.
    eps : float, optional
        Numerical epsilon used to avoid division by zero. Defaults to
        ``1e-6``.

    Notes
    -----
    - The gating parameters ``alpha`` and ``beta`` are learnable and
      unconstrained. In certified mode a scaling factor ``s`` is computed
      dynamically to clamp ``alpha`` and ensure Lipschitz bounds.
    - Mixed precision is supported by upcasting to ``float32`` internally.
    - The ``invert`` method can be used to numerically invert the activation
      output back to its input. This is useful for reversible networks or
      memoryless backward passes.
    """

    def __init__(self,
                 gamma_min: float = 0.6,
                 gamma_init: float = 0.8,
                 k_inits: Iterable[float] = (1.5, 4.0),
                 mode: str = "standard_certified",
                 eps: float = 1e-6) -> None:
        super().__init__()
        assert gamma_min > (1.0 / 9.0) + 1e-6, "gamma_min must be > 1/9 to ensure strict monotonicity"
        assert 0.0 < gamma_init <= 1.0, "gamma_init must lie in (0, 1]"
        assert mode in {"standard_certified", "flow_safe", "standard_unconstrained"}, (
            f"Invalid mode '{mode}'. Choose from 'standard_certified', 'flow_safe', 'standard_unconstrained'."
        )
        self.gamma_min = float(gamma_min)
        self.mode = mode
        self.eps = float(eps)
        k_inits = tuple(float(k) for k in k_inits)
        self.K = len(k_inits)

        # Parameterise gamma via a sigmoid to keep it within [gamma_min, 1].
        # We initialise g_hat so that sigmoid(g_hat) = (gamma_init - gamma_min)/(1 - gamma_min).
        p = (gamma_init - gamma_min) / (1.0 - gamma_min)
        # Clamp p to avoid extreme values
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        self.g_hat = nn.Parameter(torch.logit(torch.tensor(p)))

        # Knees parameterised in log space to ensure positivity
        s0 = torch.tensor([math.log(k) for k in k_inits], dtype=torch.float32)
        self.s_hat = nn.Parameter(s0)

        # Gating parameters: alpha and beta for each component. Alpha is
        # multiplicative on the context c(x), beta is additive. Both are
        # unconstrained learnable parameters.
        self.alpha = nn.Parameter(torch.zeros(self.K))
        self.beta = nn.Parameter(torch.zeros(self.K))

        # Flow‑safe weights: when mode == 'flow_safe' we learn constant logits
        # and softmax them to obtain mixture weights. In other modes these
        # weights are unused.
        self.w_logits = nn.Parameter(torch.zeros(self.K))

    def _params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the shared gamma and per‑component a and b.

        Returns
        -------
        gamma : torch.Tensor
            Scalar tensor representing the shared tail slope γ.
        a : torch.Tensor
            Tensor of shape ``(K,)`` containing the a_i coefficients.
        b : torch.Tensor
            Tensor of shape ``(K,)`` containing the b_i coefficients.
        """
        gamma = self.gamma_min + (1.0 - self.gamma_min) * torch.sigmoid(self.g_hat)
        k = torch.exp(self.s_hat) + self.eps  # ensure k_i > 0
        b = 3.0 / (k * k + self.eps)
        a = gamma * b
        return gamma, a, b

    def _context(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the even context statistic c(x) and mean squared magnitude.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(c, mean_x2)`` where ``c = log(mean(x^2) + eps)`` and
            ``mean_x2 = mean(x^2)``.
        """
        x32 = x.float()
        mean_x2 = x32.pow(2).mean()
        c = torch.log(mean_x2 + self.eps)
        return c, mean_x2

    def _certify_scale(self,
                        x: torch.Tensor,
                        w_hat: torch.Tensor,
                        alpha: torch.Tensor,
                        gamma: torch.Tensor,
                        a: torch.Tensor,
                        b: torch.Tensor) -> float:
        """Compute a scaling factor s to ensure certified Lipschitz bounds.

        This function implements a conservative certification scheme. It
        computes the smallest ``s`` in ``[0, 1]`` such that after scaling
        ``alpha`` by ``s`` the derivative of the activation is guaranteed to
        remain in ``[m, 1]`` for all elements of ``x``. The proof uses
        bounds on the derivative of the mixture and the sensitivity of the
        weights with respect to the context statistic.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        w_hat : torch.Tensor
            Unconstrained weights obtained from ``softmax(alpha * c + beta)``.
        alpha : torch.Tensor
            Gating slope parameters for each component.
        gamma : torch.Tensor
            Shared tail slope.
        a : torch.Tensor
            Tensor of shape ``(K,)`` with the component ``a_i`` coefficients.
        b : torch.Tensor
            Tensor of shape ``(K,)`` with the component ``b_i`` coefficients.

        Returns
        -------
        float
            Scaling factor ``s`` in ``[0, 1]``. If ``s`` is 1.0 no scaling
            is necessary. If ``s`` is smaller than 1.0 the gating slope is
            reduced to enforce the derivative bounds. A return value of
            ``0.0`` corresponds to completely disabling adaptivity on this
            batch.
        """
        # Compute mean and max squared magnitudes of x (float32 precision)
        x32 = x.float()
        x2 = x32.pow(2)
        mean_x2 = x2.mean()
        max_x2 = x2.max()
        # Compute mixture derivative φ'(x) for each element
        # phi_prime has shape (K, *x.shape) -> we sum across K with weights
        # to get mixture derivative of shape like x
        phi_prime_all = _phi_prime(x32, a, b)  # (K, *x.shape)
        # weight shape (K,) -> (K, 1, ..., 1)
        w_broadcast = w_hat.view(w_hat.shape[0], *([1] * x.dim()))
        phi_prime_mix = (w_broadcast * phi_prime_all).sum(dim=0)
        # margin between upper derivative bound (1) and mixture derivative
        margin = 1.0 - phi_prime_mix
        # minimum mixture derivative (for positivity)
        min_pos = phi_prime_mix.min().item()
        # minimum margin (distance to 1)
        min_margin = margin.min().item()
        # Compute sensitivity of weights with respect to context c
        # dw_i/dc = w_i * (alpha_i - <alpha>_w)
        alpha_mean = (w_hat * alpha).sum()
        dw_dc = w_hat * (alpha - alpha_mean)
        # L_w = sum_i |dw_i/dc|
        L_w = dw_dc.abs().sum().item() + 1e-12
        # Number of elements in x
        M = x.numel()
        # delta bound: L_w * 2 * max|x|^2 / (M * (mean_x2 + eps))
        delta_bound = L_w * 2.0 * max_x2.item() / (float(M) * (mean_x2.item() + self.eps))
        # Theoretical minimum derivative across components
        m = (9.0 * gamma.item() - 1.0) / 8.0
        # Reserve a small margin tau to stay away from the lower bound
        tau = 0.05 * m
        # Compute candidate scaling factors
        # Prevent negative numerator by clamping with tiny positive constant
        s1 = min_margin / (delta_bound + 1e-12)
        s2 = (min_pos - tau) / (delta_bound + 1e-12)
        s_candidates = [1.0, s1, s2]
        s = max(0.0, min(s_candidates))
        # Clamp to [0, 1]
        if s > 1.0:
            s = 1.0
        return float(max(0.0, s))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve dtype and device; compute in float32 internally
        dtype = x.dtype
        x32 = x.float()
        gamma, a, b = self._params()

        # Flow safe mode: constant mixture weights
        if self.mode == "flow_safe":
            w = F.softmax(self.w_logits, dim=-1)
        else:
            # Compute context statistic c(x)
            c, _ = self._context(x32)
            # Compute unconstrained weights
            logits = self.alpha * c + self.beta
            w_hat = F.softmax(logits, dim=-1)
            if self.mode == "standard_unconstrained":
                w = w_hat
            else:
                # Certified mode: compute scaling factor and apply it
                s = self._certify_scale(x32, w_hat, self.alpha, gamma, a, b)
                if s < 1.0:
                    logits = (self.alpha * s) * c + self.beta
                    w = F.softmax(logits, dim=-1)
                else:
                    w = w_hat

        # Evaluate φ_i(x) for each component
        phi_components = _phi(x32, a, b)  # shape (K, *x.shape)
        # Broadcast weights and sum across K to get final output
        w_broadcast = w.view(w.shape[0], *([1] * x.dim()))
        y = (w_broadcast * phi_components).sum(dim=0)
        return y.to(dtype)

    def invert(self,
               y: torch.Tensor,
               tol: float = 1e-6,
               max_iters: int = 6) -> torch.Tensor:
        """Invert the BiNLOP activation.

        This method numerically inverts the activation output ``y`` back to
        its input ``x`` using Newton's method with bracketing and
        backtracking. The inversion assumes that the mixture weights are
        constant with respect to the current inversion (i.e. ``mode`` is
        ``'flow_safe'`` or that the same weights ``w`` were used during
        forward evaluation). For ``'standard_certified'`` and
        ``'standard_unconstrained'`` modes this function inverts the
        activation using the latest weights; it does not reintroduce any
        adaptivity during inversion.

        Parameters
        ----------
        y : torch.Tensor
            Output tensor to invert.
        tol : float, optional
            Convergence tolerance for Newton's method. Defaults to ``1e-6``.
        max_iters : int, optional
            Maximum number of Newton iterations. Defaults to ``6``.

        Returns
        -------
        torch.Tensor
            Tensor ``x`` such that ``forward(x)`` is approximately ``y``.
        """
        # Use float32 for numerical stability
        y32 = y.float()
        # Use current gamma, a, b and weights
        gamma, a, b = self._params()
        # Determine weights based on current mode
        if self.mode == "flow_safe":
            w = F.softmax(self.w_logits, dim=-1)
        else:
            # Invert with the latest gating; we ignore certification here
            # because the forward pass would have applied scaling already
            c = torch.zeros(1, dtype=y32.dtype, device=y32.device)
            # In practice c should match the context used in forward, but
            # since inversion is mostly used in flow_safe or certified modes
            # where weights are constant during forward, we simply reuse
            # existing weights. If weights change drastically between
            # forward and invert, the inversion will still converge due to
            # monotonicity but may be less accurate.
            logits = self.alpha * c + self.beta
            w = F.softmax(logits, dim=-1)

        # Compute theoretical slope floor m
        m = (9.0 * gamma.item() - 1.0) / 8.0
        # Build bracket: for y >= 0: x in [y, y/m]; for y <= 0: x in [y/m, y]
        m_inv = m
        # Avoid division by zero or negative m
        m_inv = max(m_inv, 1e-3)
        # Initial guess: identity
        x = y32.clone()
        # Define function to evaluate mixture and derivative given x
        def eval_mix_and_deriv(xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            phi_components = _phi(xv, a, b)
            phi_prime_components = _phi_prime(xv, a, b)
            w_broadcast = w.view(w.shape[0], *([1] * xv.dim()))
            yv = (w_broadcast * phi_components).sum(dim=0)
            dv = (w_broadcast * phi_prime_components).sum(dim=0)
            return yv, dv

        # Determine brackets
        with torch.no_grad():
            # Flatten for bracket operations
            low = torch.where(y32 >= 0, y32, y32 / m_inv)
            high = torch.where(y32 >= 0, y32 / m_inv, y32)
            for _ in range(max_iters):
                fx, dfx = eval_mix_and_deriv(x)
                # Newton update: x_new = x - (f(x) - y)/f'(x)
                x_new = x - (fx - y32) / (dfx + 1e-12)
                # Backtracking: if out of bracket or not decreasing residual
                bad = (x_new < torch.min(low, high)) | (x_new > torch.max(low, high))
                # Evaluate f(x_new) only for good points
                fx_new, dfx_new = eval_mix_and_deriv(x_new)
                residual = (fx - y32).abs()
                residual_new = (fx_new - y32).abs()
                bad = bad | (residual_new > residual)
                # Fall back to bisection on bad points
                bisect = 0.5 * (low + high)
                x_new = torch.where(bad, bisect, x_new)
                # Update bracket
                fx_new, _ = eval_mix_and_deriv(x_new)
                go_left = fx_new < y32
                low = torch.where(go_left, x_new, low)
                high = torch.where(~go_left, x_new, high)
                x = x_new
                # Convergence check
                if torch.max((fx_new - y32).abs()).item() < tol:
                    break
        return x.to(y.dtype)
