""" random utilities """
import numpy as np
import torch

# pylint: disable=no-member
def np_unique_randint(low, high, size, overflow=1.2):
    """ returns a unique set of random ints
    Args
        low         (int)
        high        (int)
        size        (int) < high - low
        overflow    (float [1.2]) > 0
    """
    assert size < high - low, "size needs to be smaller than range"
    assert overflow > 1
    samples = np.unique(np.random.randint(low, high, int(size*overflow)))
    num_samples = len(samples)
    if num_samples < size:
        return np_unique_randint(low, high, size, overflow*1.5)

    excess = num_samples - size
    if not excess:
        return samples

    i = np.random.randint(0, size-excess)
    return np.concatenate([samples[0:i], samples[i+excess:]])


def torch_unique_randint(low, high, size, overflow=1.2):
    """ returns a unique set of random ints
    Args
        low         (int)
        high        (int)
        size        (int) < high - low
        overflow    (float [1.2]) > 0
    """
    assert size < high - low, "size needs to be smaller than range"
    assert overflow > 1
    samples = torch.unique(torch.randint(low, high, (int(size*overflow),)))
    num_samples = len(samples)
    if num_samples < size:
        return torch_unique_randint(low, high, size, overflow*1.5)

    excess = num_samples - size
    if not excess:
        return samples

    _i = torch.randint(0, size-excess, (1,))
    return torch.cat([samples[0:_i], samples[_i +excess:]])
