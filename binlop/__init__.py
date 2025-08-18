# flake8: noqa

"""
BiNLOP: Bipolar Non‑linear Operator activation functions.

This package provides a collection of activation functions based on the
BiNLOP design. The core algorithm is implemented in a framework‑agnostic
fashion and wrapped for PyTorch and TensorFlow. See the module
``torch_binlop`` for the PyTorch implementation and ``tensorflow_binlop``
for the TensorFlow implementation.

Example usage with PyTorch::

    from binlop.torch_binlop import Binlop
    act = Binlop(mode='standard_certified')
    x = torch.randn(16, 128)
    y = act(x)

The :mod:`binlop.version` module exposes the package version via the
``__version__`` attribute.
"""

from .version import __version__

# Expose the main activation classes to the top level for convenience.
try:
    from .torch_binlop import Binlop as TorchBinlop  # noqa: F401
except Exception:
    # torch is an optional dependency; ignore import errors
    TorchBinlop = None  # type: ignore

try:
    from .tensorflow_binlop import Binlop as TensorFlowBinlop  # noqa: F401
except Exception:
    TensorFlowBinlop = None  # type: ignore

__all__ = ["__version__", "TorchBinlop", "TensorFlowBinlop"]