import torch
from koreto import unsqueeze_to, extend_to


# pylint: disable=no-member
def test_unsqueeze():
    x = torch.randn(3,2)
    y = torch.ones(5,4,3,2,6)

    _dif = y.ndim - x.ndim
    _x = unsqueeze_to(x, y)
    assert _x.ndim == y.ndim
    assert list(_x.shape[:_dif]) == [1]*_dif, f"unsqueeze {x.shape} to {_x.shape}"

    y = torch.ones(5,4,2, device='cuda', dtype=torch.float16)

    _dif = y.ndim - x.ndim
    _x = unsqueeze_to(x, y)
    assert _x.ndim == y.ndim
    assert list(_x.shape[:_dif]) == [1]*_dif, f"unsqueeze {x.shape} to {_x.shape}"
    assert _x.dtype == y.dtype
    assert _x.device == y.device

    y = 5
    _dif = y - x.ndim
    _x = unsqueeze_to(x, y)
    assert _x.ndim == y
    assert list(_x.shape[:_dif]) == [1]*_dif, f"unsqueeze {x.shape} to {_x.shape}"


def test_extend():
    x = torch.randn(3,2)
    y = torch.ones(5,4,3,2,6)

    _dif = y.ndim - x.ndim
    _x = extend_to(x, y)
    assert _x.ndim == y.ndim
    assert list(_x.shape[x.ndim:]) == [1]*_dif, f"extend {x.shape} to {_x.shape}"

    y = torch.ones(5,4,2, device='cuda', dtype=torch.float16)

    _dif = y.ndim - x.ndim
    _x = extend_to(x, y)
    assert _x.ndim == y.ndim
    assert list(_x.shape[x.ndim:]) == [1]*_dif, f"extend {x.shape} to {_x.shape}"
    assert _x.dtype == y.dtype
    assert _x.device == y.device

    y = 5
    _dif = y - x.ndim
    _x = extend_to(x, y)
    assert _x.ndim == y
    assert list(_x.shape[x.ndim:]) == [1]*_dif, f"extend {x.shape} to {_x.shape}"
