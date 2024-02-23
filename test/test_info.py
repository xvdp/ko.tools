import torch
from koreto.info import tensor_flat2

def test_tensor_flat2():
    x = torch.zeros(2,3,4,5)
    assert tensor_flat2(x).shape == (2, 3*4*5), f"{tensor_flat2(x).shape} !== (2, 3*4*5)"
    assert tensor_flat2(x, 0, 0).shape == (1, 2*3*4*5), f"{tensor_flat2(x, 0).shape} == (1, 2*3*4*5)"
    assert tensor_flat2(x.view(-1)).shape == (1, 2*3*4*5), f"{tensor_flat2(x, 0).shape} == (1, 2*3*4*5)"
    assert tensor_flat2(x, 0, 2).shape == (2*3, 4*5), f"{tensor_flat2(x, 0, 2).shape} == (2*3, 4*5)"
    assert tensor_flat2(x, 0, 3).shape == (2*3*4, 5), f"{tensor_flat2(x, 0, 3).shape} == (2*3*4, 5)"
    assert tensor_flat2(x, 3, 1).shape == (5, 2*3*4), f"{tensor_flat2(x, 3, 1).shape} == (5, 2*3*4)"
    assert tensor_flat2(x, 3, 2).shape == (5*2, 3*4), f"{tensor_flat2(x, 3, 2).shape} == (5*2, 3*4)"
    assert tensor_flat2(x, 1, 1).shape == (3, 2*4*5), f"{tensor_flat2(x, 1, 1).shape }== (3, 2*4*5)"
    assert tensor_flat2(x, 2, 1).shape == (4, 2*4*5), f"{tensor_flat2(x, 1, 2).shape} == (4, 2*4*5)"
    assert tensor_flat2(x, 2, 4).shape == tensor_flat2(x, 1, 4).shape == (2*3*4*5,1), f"{tensor_flat2(x, 1, 4).shape} == (2*3*4*5,1)"
