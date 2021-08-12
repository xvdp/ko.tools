""" image / camera intrinsics functions
wip. options built to match various projects

"""
import numpy as np
import torch
from .utils import ObjDict

# pylint: disable=no-member
def undistorted_rays(xy, xy0, ks, ps):

    rad = torch.sum(xy**2, axis=-1, keepdims=True)
    dist = 1.0 + rad * (ks[0] + rad * (ks[1] + ks[2] * rad))

    fxy = dist * xy + 2 * ps * torch.prod(xy, dim=-1, keepdim=True) + ps.flip(-1) * (rad + 2*xy**2) - xy0

    # dist'
    dist_xy = 2.0*xy*(ks[0] + rad*(2.0*ks[1] + 3.0*ks[2]*rad))

    # fx' over x and y
    _62 = torch.tensor([6,2], dtype=xy.dtype, device=xy.device)
    fx_xy = dist_xy.mul(xy[..., 0:1]) + xy.flip(-1).mul(ps[0]*2) + xy.mul(_62*ps[1])
    fx_xy[..., 0].add_(dist.squeeze())

    # fy' over x and y
    fy_xy = dist_xy.mul(xy[..., 1:2]) + xy.flip(-1).mul(ps[1]*2) + xy.mul(_62.flip(-1)*ps[0]) 
    fy_xy[..., 1].add_(dist.squeeze())

    return fxy, fx_xy, fy_xy

def pixels_to_rays(pixels, center, focal, radial=None, tangential=None, ratio=1., skew=0, iters=10, z=1.0,
                   rotation=None, normalize=True):
    """ image pixels to camera rays
    Args
        pixels      tensor (h,w,2)
        center      tensor (2)          [cx,cy] principal_point
        focal       float | tensor(2)   [fx,fy] focal_
        radial      tensor  [k1,k2,k3]  # if None: dont undistort
        tangential  tensor  [p1,p2]     # it None: dont undistort
    optional Args
        ratio       float [1.] pixel aspect ratio
        z           float [1.]: if -1 flip coordinates z and y
        iters       int [10] ray optimization iterations
        rotation    torch.tensor (3,3) [None], camera extrinsics
        normalize   bool [True]  normalize resulting rays
    """
    _as_tensor = {"device":pixels.device, "dtype":pixels.dtype}
    xy = pixels.sub(center).div(focal).mul(torch.tensor([1., z/ratio], **_as_tensor))

    if skew:
        if isinstance(focal, torch.tensor) and focal.dim() == 2:
            focal = focal[-1]
        xy[..., 0].sub_(xy[..., 1], alpha=skew/focal)

    if radial is not None and tangential is not None and (any(radial) or any(tangential)):
        xy0 = xy.clone().detach()
        for _ in range(iters):
            fxy, fx_xy, fy_xy = undistorted_rays(xy, xy0, radial, tangential)

            denom = (fy_xy[...,0] * fx_xy[...,1] - fx_xy[...,0] * fy_xy[...,1]).unsqueeze(-1)

            step = torch.ones_like(xy)
            step[...,0].mul_(fxy[...,0] * fy_xy[...,1] -  fxy[...,1] * fx_xy[...,1])
            step[...,1].mul_(fxy[...,1] * fx_xy[...,0] -  fxy[...,0] * fy_xy[...,0])
            step.div_(denom).where(denom.abs() > 1e-8, torch.zeros_like(xy))
            xy.add_(step)

    shape = list(xy.shape[:2]) + [1]
    out = torch.cat((xy, z * torch.ones(shape, **_as_tensor)), dim=-1)

    if rotation is not None: # extrinsics
        out = (rotation @ out.unsqueeze(-1)).squeeze(-1)
    if normalize:
        out = out.div(out.norm(dim=-1, keepdim=True))
    return out

def rotate_rays(rays, rotation):
    """
    Args
        rays        tensor (...,3)
        rotation    tensor (3,3) # if row major: pass .T
    """
    return (rotation @ rays.unsqueeze(-1)).squeeze(-1)


def transform_points(points, position=None, rotation=None):
    """
    Args
        points      tensor (...,3)
        position    tensor (1, 3) 
        rotation    tensor (3, 3) if row major: pass .T
    """
    if position is not None:
        points = points - position
    if rotation is not None:
        points = (rotation @ points.T).T
    return points


def points_to_pixels(points, center, focal, radial=None, tangential=None, ratio=1., skew=0, position=None, rotation=None):
    """Projects a 3D point (x,y,z) to a pixel position (x,y).
     Args
        points      tensor (...,3)
        position    tensor (1, 3)
        rotation    tensor (3, 3) if row major: pass .T

        center      tensor (2)          [cx,cy] principal_point
        focal       float | tensor(2)   [fx,fy] focal_

        radial      tensor  [k1,k2,k3]  # if None: dont undistort
        tangential  tensor  [p1,p2]     # it None: dont undistort
    """
    batch_shape = points.shape[:-1]
    points = points.view(-1, 3)
    points = transform_points(points, position, rotation)
    _as_tensor = {"device":points.device, "dtype":points.dtype}
    ratio = torch.tensor([1., ratio], **_as_tensor)

    xy = points[..., 0:2]/points[..., 2:3]
    if radial is not None and tangential is not None and (any(radial) or any(tangential)):
    
        # Radial distortion.
        r2 = torch.sum(xy**2, axis=-1, keepdims=True)
        xy.mul_(1.0 + r2 * (radial[0] + r2 * (radial[1] + radial[2] * r2)))

        # Tangential distorsion
        xy.add_(2.0 * tangential * xy.prod(axis=-1) + tangential.filp(-1) * (r2 + 2.0 * xy**2))

        # Map to image plane
        xy[..., 0].add_(xy[..., 1].mul(skew))
        xy.mul_(ratio*focal).add_(center)

    return xy.reshape((*batch_shape, 2))

        # torch.tensor([1., ratio], **_as_tensor)
        # pixels = center - xy

        # xy = pixels.sub(center).div(focal).mul(torch.tensor([1., z/ratio], **_as_tensor))

    # Get normalized local pixel positions.
    # x = points[..., 0] / points[..., 2]
    # y = points[..., 1] / points[..., 2]
    # r2 = x**2 + y**2



    # if radial is not None and tangential is not None and (any(radial) or any(tangential)):
    #     # Apply radial distortion.
    #     distortion = 1.0 + r2 * (radial[0] + r2 * (radial[1] + radial[2] * r2))

        # Apply tangential distortion.
        # x_times_y = x * y
        # x = x * distortion + 2.0 * tangential[0] * x_times_y + tangential[1] * (r2 + 2.0 * x**2)
        # y = y * distortion + 2.0 * tangential[1] * x_times_y + tangential[0] * (r2 + 2.0 * y**2)

        # # Map the distorted ray to the image plane and return the depth.
        # pixel_x = focal * x         + center[0] + skew * y
        # pixel_y = focal * y * ratio + center[1]

        # pixels = torch.stack([pixel_x, pixel_y], axis=-1)

    # return pixels.reshape((*batch_shape, 2))


def copy_vals(fro, to, repeat=False):
    """ copies into tensor
    Args
        to      int, float, torch.Tensor
        fro     int, float, torch.Tensor, np.ndarray
    """
    _grad = to.requires_grad
    if isinstance(to, (int, float)):
        to = type(to)(fro)
        print("intfloat")

    elif isinstance(to, torch.Tensor):
        _asto = {"dtype":to.dtype, "device":to.device}
        if isinstance(fro, (int, float)):
            if repeat:
                to[:] = to.mul(0).add(fro)
            else:
                to[0] = fro
        elif isinstance(fro, torch.Tensor) and fro.shape == to.shape and not to.requires_grad:
            to = fro.clone().detach().to(**_asto)
        elif isinstance(fro, np.ndarray) and tuple(to.shape) == tuple(fro.shape):
            to = torch.as_tensor(fro, **_asto)
        else:
            if isinstance(fro, np.ndarray):
                fro = fro.reshape(-1)
            _shape = to.shape
            to = to.view(-1)
            _len = len(fro) if repeat else min(len(to), len(fro))
            for i in range(_len):
                to[i] = fro[i%len(fro)].item()
            to = to.view(_shape)

    to.requires_grad = _grad


class Camera:
    """ camera intrinsics and extrinsics in pytorch
    """
    def __init__(self, **kwargs):
        _x = 100.0
        self.center = torch.tensor([_x/2.,_x/2]) # cx,cy
        self.focal = torch.tensor([_x, _x]) # fx, fy
        self.radial = torch.zeros(3) # k1, k2, k3
        self.tangential = torch.zeros(2) # p1, p2
        self.height = _x
        self.width = _x

        self.position = torch.zeros(3)
        self.rotation = torch.eye(3, 3)

        self.from_keyvalues(**kwargs)

    def to(self, **kwargs):
        """
            kwargs
                device  (str) ['cuda' | 'cpu']
                dtype   (torch.dtype)
        """
        _to = {k:kwargs[k] for k in kwargs if k in ("device", "dtype")}

        for x in self.__dict__:
            if isinstance(self.__dict__[x], torch.Tensor):
                self.__dict__[x].to(**_to)


    def from_keyvalues(self, **kwargs):
        """ intrinsics can be either in the form
                * radial=[], tangential=[], center=[], focal=[]
                * k1=,k2=,k3=,p1=,p2=,cx=,cy=,fx=,fy=
                height=, width=,
            extrinsics
                position=[], rotation=[]
        """
        for key in kwargs:
            if key in self.__dict__:
                copy_vals(kwargs[key], self.__dict__[key])
            elif key == "k1":
                self.radial[0] = kwargs[key]
            elif key == "k2":
                self.radial[1] = kwargs[key]
            elif key == "k3":
                self.radial[3] = kwargs[key]
            elif key == "p1":
                self.tangential[0] = kwargs[key]
            elif key == "p2":
                self.tangential[1] = kwargs[key]
            elif key == "cx":
                self.center[0] = kwargs[key]
            elif key == "cy":
                self.center[1] = kwargs[key]
            elif key == "fx":
                self.focal[0] = kwargs[key]
            elif key == "fy":
                self.focal[1] = kwargs[key]

    def to_keyvalues(self):
        """ outputs obj dict in opencv format
        """
        out = ObjDict()
        out.k1 = self.radial[0]
        out.k2 = self.radial[1]
        out.k3 = self.radial[2]
        out.p1 = self.tangential[0]
        out.p2 = self.tangential[1]
        out.cx = self.center[0]
        out.cy = self.center[1]
        out.fx = self.focal[0]
        out.fy = self.focal[1]
        out.height = self.height
        out.width = self.width
        return out

    def from_colmap(self, cam=None, img=None):
        """
        Args
            cam     pycolmap.scene.cameras[i]
            img     pycolmap.scene.images[j]
        """
        if cam is not None:
            self.center[0] = cam.cx
            self.center[1] = cam.cy
            self.focal[0] = cam.fx
            self.focal[1] = cam.fy
            self.radial[0] = cam.k1
            self.radial[1] = cam.k2
            self.tangential[0] = cam.p1
            self.tangential[1] = cam.p2
            self.height = cam.height
            self.width = cam.width

        if img is not None:
            copy_vals(img.t, self.position)
            copy_vals(img.R(), self.rotation)

    def from_colmap_scene(self, scene, index):
        """
        Args
            scene   (pycolmap scene)
            index   (int) scene.images[index]
        """
        self.from_colmap(cam=scene.cameras[scene.images[index].camera_id],
                         img=scene.images[index])
