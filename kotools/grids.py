""" mesh grids
"""
import numpy as np
from kotools import WITH_TORCH
if WITH_TORCH:
    import torch

# pylint: disable=no-member
def mgrid(shape, dtype="float32", shift=0.5, flip_columns=True, layout=1, form="torch"):
    """ fast nd mgrid: not transposing means contiguity requires no fixing
    Args
        shape           (tuple, list) any number of dimensions
        dtype           torch.dtype [torch.float32]
        shift           float [0.5]
        flip_columns    bool [True]: col[0] corresponds to shape[-1]
        layout          int [1]: [..., dims] 0: [dims, ...]
    """
    if not WITH_TORCH or form[0] == "n":
        return np_mgrid(shape, dtype=dtype, shift=shift, flip_columns=flip_columns, layout=layout)

    dtype = dtype if isinstance(dtype, torch.dtype) else torch.__dict__[dtype]
    with torch.no_grad():
        _layout = (*shape, len(shape)) if layout else (len(shape), *shape)
        out = torch.ones(_layout, dtype=dtype)

        for i, side in enumerate(shape):
            view = [1] * len(shape)
            view[i] = side
            col = i if not flip_columns else len(shape)-i-1
            if layout:
                out[..., col].mul_(torch.arange(shift, side+shift, 1,
                                                dtype=dtype).view(*view))
            else:
                out[col, ...].mul_(torch.arange(shift, side+shift, 1,
                                                dtype=dtype).view(*view))
    return out

def mgrid_pos(idx, shape, shift=0.5, dtype="float32", flip_columns=True, layout=1, form="torch"):
    """ return dtype [float32] mesh grid positions for input flat indices
        Args:
            idx             flat indices of mgrid position
            shape           tuple
            shift           float [0.5] pixel center
            dtype           torch.dtype [torch.float32]
            flip_columns    bool[True]  reverse column order
            layout          int [1]: [N, dims] 0: [dims, N]
    """
    if not WITH_TORCH or form[0] == "n":
        return np_mgrid_pos(idx, shape, shift=shift, dtype=dtype, flip_columns=flip_columns, layout=layout)

    dtype = dtype if isinstance(dtype, torch.dtype) else torch.__dict__[dtype]
    idx = torch.as_tensor(idx, dtype=dtype)
    _layout = (len(idx), len(shape)) if layout else (len(shape), len(idx))
    shape = torch.asarray(shape)
    out = torch.ones(_layout, dtype=dtype)

    for i, side in enumerate(shape):
        col = i if not flip_columns else len(shape)-i-1
        view = [1] * len(shape)
        view[i] = side

        if layout:
            out[..., col].mul_((idx//torch.prod(shape[i+1:]))%shape[i])
        else:
            out[col, ...].mul_((idx//torch.prod(shape[i+1:]))%shape[i])
    return out + shift

##
# numpy versions
def np_mgrid(shape, dtype="float32", shift=0.5, flip_columns=True, layout=1):
    """ fast nd mgrid: not transposing means contiguity requires no fixing
    Args
        shape           (tuple, list) any number of dimensions
        dtype           np.dtype [np.float32]
        shift           float [0.5]
        flip_columns    bool [True]: col[0] corresponds to shape[-1]
        layout          int [1]: [..., dims] 0: [dims, ...]
    """
    dtype = dtype if isinstance(dtype, np.dtype) else np.__dict__[dtype]
    _layout = (*shape, len(shape)) if layout else (len(shape), *shape)
    out = np.ones(_layout, dtype=dtype)

    for i, side in enumerate(shape):
        view = [1] * len(shape)
        view[i] = side
        col = i if not flip_columns else len(shape)-i-1
        if layout:
            out[..., col] *= np.arange(shift, side+shift, 1, dtype=dtype).reshape(*view)
        else:
            out[col, ...] *= np.arange(shift, side+shift, 1, dtype=dtype).reshape(*view)
    return out

def np_mgrid_pos(idx, shape, shift=0.5, dtype="float32", flip_columns=True, layout=1):
    """ return mesh grid positions for input flat indices
        Args:
            idx             flat indices of mgrid position
            shape           tuple
            shift           float [0.5] pixel center
            dtype           np.dtype [np.float32]
            flip_columns    bool[True]  reverse column order
            layout          int [1]: [N, dims] 0: [dims, N]
    """
    dtype = dtype if isinstance(dtype, np.dtype) else np.__dict__[dtype]
    idx = np.asarray(idx, dtype=dtype)
    _layout = (len(idx), len(shape)) if layout else (len(shape), len(idx))
    shape = np.asarray(shape)
    out = np.ones(_layout, dtype=dtype)

    for i, side in enumerate(shape):
        col = i if not flip_columns else len(shape)-i-1
        view = [1] * len(shape)
        view[i] = side

        if layout:
            out[..., col] *= ((idx//np.prod(shape[i+1:]))%shape[i])
        else:
            out[col, ...] *= ((idx//np.prod(shape[i+1:]))%shape[i])       
    return out + shift
