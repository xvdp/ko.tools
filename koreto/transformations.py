"""
numpy / torch transformations with inputs of shapes [..., C]
"""
from typing import Union
import math
import numpy as np
import torch
Vector = Union[np.ndarray, torch.Tensor]

# pylint: disable=no-member
# pylint: disable=not-callable

def mean_dist(p, size=1024, k=3):
    """ KISS method to find mean distance to closest points
    No good if too many points.
    """
    N, C = p.shape
    steps = math.ceil(N/size)
    j = dict(axis=-1)
    return torch.cat([
        ((p[i*size: (i+1)*size][:, None] - p)**2).sum(**j).sort(**j)[0][:, 1:k+1].mean(**j)
         for i in range(steps)
    ])

def estimate_points_normals(points: torch.Tensor, k: int = 10) -> torch.Tensor:
    """ estimate point normals on local point cloud 
    No good if 4 * len(points)**2 > torch.cuda.get_device_properties(0).total_memory 
    Args
        points  (Tensor) shape(N, 3)
        k       (int [10])
    """
    _, idcs = torch.cdist(points, points).topk(k + 1, largest=False)
    return fit_plane_normal(points[idcs[:, 1:]]) # [N, k, 3]


def fit_plane_normal(points: Vector) -> Vector:
    """Fit a plane to the points and return normal or normals

    Args
        points  ndarray or tensor shape(..., N, 3)
            if ndim == 3, computes multiple normals

    Could be done with leastsq  or eigen vectors
    Faster than eigen vectors
        l_pts = points - points.mean(axis=0)
        cov_matrix = l_pts.t() @ l_pts/len(l_pts)
        eigvals, eigvecs  = torch.linalg.eigh(cov_matrix, UPLO='U')
        return eigvecs[:, 0]
    """
    linalg = torch.linalg if torch.is_tensor(points) else np.linalg
    _, _, vh = linalg.svd(points - points.mean(axis=-2, keepdim=True))
    return vh[..., 2, :]


def rot_from_vectors(v0: np.ndarray, v1: Union[str, np.ndarray]) -> np.ndarray:
    """ 3x3 rotation from 2 vectors
    Args
        normal  (ndarray (3))
        v1    (str in x,y,z or ndarray(3))
    TODO fix: compute multiple rotations 
    """
    if isinstance(v1, str):
        v1 = v1.lower()
        assert v1 in ('x', 'y', 'z'), f"invalid axis {v1}"
        v1 = {'x':np.asarray([1.,0.,0.]),
                'y':np.asarray([0.,1.,0.]),
                'z':np.asarray([0.,0.,1.])}[v1]
    assert isinstance(v1, np.ndarray) and v1.shape == (3,), f"invalidad axis {v1}"
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    if np.allclose(v0, v1):
        return np.eye(3)

    cross = np.cross(v0, v1)
    dot = np.dot(v0, v1)
    skewmat = np.array([[       0,  -cross[2],   cross[1]],
                        [cross[2],          0,  -cross[0]],
                        [-cross[1],  cross[0],   0]])
    return np.eye(3) + skewmat + np.dot(skewmat, skewmat)/(1 + dot)


def rotate(points: np.ndarray, rotmat: np.ndarray, atol: int = 8) -> np.ndarray:
    """ rotate points to arbitrary axis
    Args
        points  (ndarray (N, 3))
        rotmat  (ndarray (3, 3))
        atol    (int [8]) round floating point errors
    """
    return np.round(np.dot(points, rotmat.T), atol)
