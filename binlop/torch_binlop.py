import torch
import torch.nn as nn
import torch.nn.functional as F

class _BiNLOPOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, k):
        # gamma, k are broadcastable to x
        y = gamma * x + (1.0 - gamma) * torch.clamp(x, -k, k)
        # Save only what backward needs for memoryless gradients
        ctx.save_for_backward(y, gamma, k)
        return y

    @staticmethod
    def backward(ctx, gy):
        y, gamma, k = ctx.saved_tensors
        # region mask from y (knees map to y=±k)
        M = (y.abs() > k).to(gy.dtype)
        signy = torch.sign(y)
        # dx
        dx = gy * (1.0 - (1.0 - gamma) * M)
        # dgamma
        clamp_y = torch.clamp(y, -k, k)
        dgamma = gy * ((y - clamp_y) / (gamma + 1e-12)) * M
        # dk
        dk = gy * (1.0 - gamma) * signy * M
        # Reduce gamma/k grads over broadcasted dims
        while dgamma.dim() > gamma.dim():
            dgamma = dgamma.sum(dim=0)
            dk = dk.sum(dim=0)
        # Sum over parameter dims that were broadcast
        for i, need_sum in enumerate(gamma.shape):
            pass  # Rely on autograd broadcasting; alternative: keep gamma/k as leaf scalars per module
        return dx, dgamma, dk


class BiNLOP2(nn.Module):
    """
    BiNLOP-2: φ(x) = γ x + (1 − γ) clamp(x, −k, k)
    Odd, monotone, bi-Lipschitz, invertible; dtype-preserving; memoryless backward.
    """
    def __init__(self, gamma_min=0.6, gamma_init=0.85, k_init=2.0, per_channel=False, num_channels=None):
        super().__init__()
        assert 0.0 < gamma_min < 1.0 and gamma_min < gamma_init < 1.0
        self.gamma_min = float(gamma_min)
        self.per_channel = per_channel
        if per_channel:
            assert num_channels and num_channels > 0
            p = (gamma_init - gamma_min) / (1 - gamma_min)
            p = float(max(min(p, 1 - 1e-6), 1e-6))
            self.g_hat = nn.Parameter(torch.logit(torch.tensor(p)).repeat(num_channels))
            self.s_hat = nn.Parameter(torch.log(torch.tensor(k_init)).repeat(num_channels))
        else:
            p = (gamma_init - gamma_min) / (1 - gamma_min)
            p = float(max(min(p, 1 - 1e-6), 1e-6))
            self.g_hat = nn.Parameter(torch.logit(torch.tensor(p)))
            self.s_hat = nn.Parameter(torch.log(torch.tensor(k_init)))

    def _params(self, x):
        gamma = self.gamma_min + (1.0 - self.gamma_min) * torch.sigmoid(self.g_hat)
        k = torch.exp(self.s_hat)
        if self.per_channel:
            if x.dim() == 4:   # NCHW
                gamma = gamma.view(1, -1, 1, 1)
                k = k.view(1, -1, 1, 1)
            elif x.dim() == 3: # NLC
                gamma = gamma.view(1, 1, -1)
                k = k.view(1, 1, -1)
        return gamma.to(x.dtype), k.to(x.dtype)

    def forward(self, x):
        gamma, k = self._params(x)
        return _BiNLOPOp.apply(x, gamma, k)

    @torch.no_grad()
    def invert(self, y):
        gamma, k = self._params(y)
        s = y.abs()
        core = s <= k
        inv_gamma = (1.0 / gamma).to(y.dtype)
        x = torch.where(core, y, y.sign() * (k + (s - k) * inv_gamma))
        return x

    def logdet(self, x):
        gamma, k = self._params(x)
        mask = (x.abs() > k)
        return (mask.to(x.dtype) * torch.log(gamma.to(x.dtype))).sum()
