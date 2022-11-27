"""@xvdp
some standard tensor utilities
"""
from typing import Union
from torch import Tensor


def _to(x, y, position=0):
    _yndim = y if isinstance(y, int) else y.ndim
    assert _yndim >= x.ndim, f"io tensor is higher dimensional than target {x.ndim} > {_yndim}"
    if position:
        _shape =  list(x.shape) + [1] * (_yndim - x.ndim)
    else:
        _shape = [1] * (_yndim - x.ndim) + list(x.shape)
    x = x.view(_shape) if x.is_contiguous() else x.reshape(_shape)
    if isinstance(y, Tensor):
        return x.to(y)
    return x

def extend_to(x: Tensor, y: Union[Tensor, str]) -> Tensor:
    """ extend trailing dimensions of tensor x to tensor shape or int y.
    Args
        x   Tensor io tensor
        y   Tensor or str, model tensor or ndims
    """
    return _to(x, y, 1)

def unsqueeze_to(x: Tensor, y: Union[Tensor, str]) -> Tensor:
    """ unsqueeze leading dimensions of tensor x to tensor shape or int y.
    Args
        x   Tensor io tensor
        y   Tensor or str, model tensor or ndims
    """
    return _to(x, y, 0)
