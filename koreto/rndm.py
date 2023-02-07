"""@xvdp
random utilities 

    rndlist(inputs) # returns random subset from list
    randint(items)  # returns random index in list, int or tensor
    randitem(items) # returns random item in sequence

"""
from typing import Any, Union
import numpy as np
import random
from koreto import WITH_TORCH
if WITH_TORCH:
    import torch
    Vector = Union[np.ndarray, torch.Tensor]
    Index = Union[int, torch.Tensor]
else:
    Vector = np.ndarray
    Index = int

# pylint: disable=no-member

def randint(items: Union[Vector, tuple, list, int], astorch: bool = False) -> Index:
    """ returns random index in list, int or tensor
    Args
        items   (list|tuple|np.ndarray|torch.Tensor)
        astorch (bool [False]) if True use torch random functions
    """
    items = items if isinstance(items, int) else len(items)
    if astorch and WITH_TORCH:
        out = torch.randint(0, items, (1,))
    else:
        out = random.randint(0, items-1)
    return out


def randitem(items: Union[Vector, tuple, list],
             astorch: bool = False,
             verbose: bool = False) -> Any:
    """ returns random item in sequence
    Args
    items   (list|tuple|np.ndarray|torch.Tensor)
    astorch (bool [False]) if True use torch random functions
    verbose (bool [False]) if True print item
    """
    item = randint(items, astorch)
    if verbose:
        print(item)
    return items[item]



def rndlist(inputs: Union[list, tuple, Vector], num: int = 1) -> list:
    """ returns random subset from list
    Args
        inputs   (iterable)
        num      (int [1]) number of elements returned
    """
    choice = np.random.randint(0, len(inputs), num)
    if isinstance(inputs, Vector):
        return inputs[choice]
    return [inputs[c] for c in choice]


def unique_randint(low, high, size, overflow=1.2, out_type="torch"):
    """ returns a unique set of random ints
    Args
        low         (int)
        high        (int)
        size        (int) < high - low
        overflow    (float [1.2]) > 1

        out_type    (str ["torch"]) | "numpy"
    """
    assert size < high - low, "size needs to be smaller than range"
    assert overflow > 1
    if not WITH_TORCH:
        out_type = "numpy"
    if out_type[0] == "n":
        return _np_unique_randint(low, high, size, overflow=1.2)

    samples = torch.unique(torch.randint(low, high, (int(size*overflow),)))
    num_samples = len(samples)
    if num_samples < size:
        return unique_randint(low, high, size, overflow*1.5)

    excess = num_samples - size
    if not excess:
        return samples

    _i = torch.randint(0, size-excess, (1,))
    return torch.cat([samples[0:_i], samples[_i +excess:]])


def _np_unique_randint(low, high, size, overflow=1.2):
    """ returns a unique set of random ints
    Args
        low         (int)
        high        (int)
        size        (int) < high - low
        overflow    (float [1.2]) > 1
    """
    samples = np.unique(np.random.randint(low, high, int(size*overflow)))
    num_samples = len(samples)
    if num_samples < size:
        return _np_unique_randint(low, high, size, overflow*1.5)

    excess = num_samples - size
    if not excess:
        return samples

    i = np.random.randint(0, size-excess)
    return np.concatenate([samples[0:i], samples[i+excess:]])
