"""@xvdp
point utilities

outlier methods
irq_keep()      keep points < 75% of distance
zscore_keep()   keep points < 3 stds of distance
"""
from typing import Union, Tuple, Optional
import numpy as np
import torch
Vector = Union[torch.Tensor, np.ndarray]


def sort_points_by_distance(points: Vector,
                            center: Optional[Vector] = None) -> Tuple[Vector, Vector]:
    """ sort points by distance to a center, or compute center if not found
    Returns (sorted indices, distance square)
    Args:
        points  (ndarray, Tensor) (..., N, 3)
        center  (ndarray, Tensor) (..., 3)  
    """
    if center is None:
        center = points.mean(axis=-2)
    dist_2 = ((points - center)**2).sum(axis=-1)
    sort_idcs = dist_2.argsort(axis=-1)
    return sort_idcs, dist_2


def zscore_keep(dists: np.ndarray, stds: float = 3) -> np.ndarray:
    """ threshod distances by stds
    Returns indices of distances < stds
    Args
        dists   ndarray, Tensor
        stds    float [3]
    """
    _op = torch if torch.is_tensor(dists) else np
    return _op.where(_op.abs((dists - dists.mean(axis=-1))/dists.std(axis=-1)) < stds)[0]


def irq_keep(dists: Vector,
             clip_far: bool = True,
             clip_near: bool = False,
             irq_expand: float = 1.5) -> Vector:
    """ threshold distances by percentile discarding those beyond the upper bound
    Returns indices
    Args
        dists       ndarray, Tensor
        clip_far    bool [True]
        clip_near   bool [False]   default keeps close points
    """
    _op = torch if torch.is_tensor(dists) else np
    if not (clip_far or clip_near): # pointless: all points.
        return _op.arange(len(dists))
    q1 = np.percentile(dists, 25)
    q3 = np.percentile(dists, 75)
    iqr = q3 - q1
    lower_bound = q1 - irq_expand * iqr
    upper_bound = q3 + irq_expand * iqr
    clip_far = (dists < upper_bound) if clip_far else True
    clip_near = (dists > lower_bound) if clip_near else True
    return _op.where(clip_far & clip_near)[0]
