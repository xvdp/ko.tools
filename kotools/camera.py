""" image / camera intrinsics functions
wip. options built to match various projects
"""
import torch

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

def pix_to_rays(pix, center, focal, radial, tangential, ratio=1., skew=0, iters=10, z=1.0, normalize=True):
    """ image pixels to camera rays
    Args
        pix         tensor (h,w,2)
        center      tensor (2)          [cx,cy] principal_point
        focal       float | tensor(2)   [fx,fy] focal_
        radial      tensor  [k1,k2,k3]  # if None: dont undistort
        tangential  tensor  [p1,p2]     # it None: dont undistort
    optional Args
        ratio       float [1.] pixel aspect ratio
        z           float [1.]: if -1 flip coordinates z and y
        iters       int [10] ray optimization iterations
        normalize   bool [True]  normalize resulting rays
    """
    _as_tensor = {"device":pix.device, "dtype":pix.dtype}
    xy = pix.sub(center).div(focal).mul(torch.tensor([1., z/ratio], **_as_tensor))

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
    if normalize:
        out.div_(out.norm(dim=-1, keepdim=True))
    return out

def rotate_rays(rays, rotation):
    """
    Args
        rays        tensor(...,3)
        rotation    tensor (3,3)
            if row major: pass .T
    """
    return (rotation @ rays.unsqueeze(-1)).squeeze(-1)
