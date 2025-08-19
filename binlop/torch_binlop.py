import torch
import torch.nn as nn
import torch.nn.functional as F


class BiNLOP(nn.Module):
    """
    BiNLOP-1: Linear-Tail Hinge
    phi(x) = x - (1 - gamma) * sign(x) * relu(|x| - k)
    Guarantees: odd, monotone, bi-Lipschitz with phi' in {1, gamma}, invertible.
    """

    def __init__(self, gamma_min=0.6, gamma_init=0.8, k_init=2.0, per_channel=False, num_channels=None):
        super().__init__()
        assert 0.0 < gamma_min < 1.0 and gamma_min < gamma_init < 1.0
        self.gamma_min = float(gamma_min)
        self.per_channel = per_channel

        p = (gamma_init - gamma_min) / (1 - gamma_min)
        p = float(max(min(p, 1 - 1e-6), 1e-6))

        if per_channel:
            assert num_channels is not None and num_channels > 0
            self.g_hat = nn.Parameter(torch.logit(torch.tensor(p)).repeat(num_channels))
            self.s_hat = nn.Parameter(torch.log(torch.tensor(k_init)).repeat(num_channels))
        else:
            self.g_hat = nn.Parameter(torch.logit(torch.tensor(p)))
            self.s_hat = nn.Parameter(torch.log(torch.tensor(k_init)))

    def _params(self, x):
        gamma = self.gamma_min + (1.0 - self.gamma_min) * torch.sigmoid(self.g_hat)
        k = torch.exp(self.s_hat)

        if self.per_channel:
            if x.dim() == 4:   # NCHW
                gamma = gamma.view(1, -1, 1, 1)
                k = k.view(1, -1, 1, 1)
            elif x.dim() == 3:  # NLC
                gamma = gamma.view(1, 1, -1)
                k = k.view(1, 1, -1)

        return gamma, k

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        gamma, k = self._params(x32)
        s = x32.abs()
        t = F.relu(s - k)
        y = x32 - (1.0 - gamma) * x32.sign() * t
        return y.to(dtype)

    @torch.no_grad()
    def invert(self, y):
        y32 = y.float()
        gamma, k = self._params(y32)
        s = y32.abs()
        core = s <= k
        inv_gamma = 1.0 / gamma
        x = torch.where(
            core,
            y32,
            y32.sign() * (k + (s - k) * inv_gamma)
        )
        return x.to(y.dtype)
