""" @xvdp   data utility classes
    ObjDict     object access dictionary
    TraceMem    ObjDict collecting memory usage
    GPUse       thin wrap of nvidia-smi
        TODO replace with nvml
    CPUse       thin wrap of psutils
"""
from typing import TypeVar, Any, Optional, Callable
import inspect
from copy import deepcopy
import os
import os.path as osp
import re
import json
import numpy as np
import yaml

from koreto import WITH_TORCH
if WITH_TORCH:
    import torch

_T = TypeVar('_T')

# pylint: disable=no-member
# pylint: disable=suppressed-message
# ###
# Dictionaries and Memory management
#
class IPP:
    """ i++ for a namespace"""
    def __init__(self, i=0):
        self.i = i
    @property
    def pp(self):
        self.i += 1
        return self.i - 1
    def p(self, j=1):
        self.i += j
        return self.i - j

class DeepClone:
    """ similar to deep copy detaching tensors to cpu
        self.out    cloned data
        self.stats  counter of classes cloned
    TODO deepclone only detaches tensors on first level, need to make recursive
    TODO write tests, this has too many failure points possible
    """
    def __init__(self, data: Any, cpu: bool = True) -> None:
        self._cpu = cpu
        self.out = None
        self.stats = {}
        self.out = self.clone(data)

    def clone(self, data: _T) -> _T:
        _type = data.__class__
        if _type not in self.stats:
            self.stats[_type] = 0
        self.stats[_type] += 1

        if isinstance(data, dict):
            return self.clone_dict(data)

        if isinstance(data, (list, tuple)):
            return self.clone_list(data)
        if WITH_TORCH and isinstance(data, torch.Tensor):
            with torch.no_grad():
                if self._cpu:
                    return data.cpu().clone().detach()
                return data.clone().detach()
        else:
            return deepcopy(data)

    def clone_dict(self, data: dict) -> dict:
        out = data.__class__()
        for k in data:
            out[k] = self.clone(data[k])
        return out

    def clone_list(self, data: list) -> list:
        """ lists and tuples - fix typing"""
        return data.__class__(self.clone(d) for d in data)

def deepclone(data: _T) -> _T :
    """ similar to deep copy detaching tensors to cpu
            out = deepclone(data)
    """
    return DeepClone(data).out

def todict(obj):
    """ convert ObjDict recursively to dict
    """
    if isinstance(obj, (list, tuple)):
        _totuple = isinstance(obj, tuple)
        obj = list(obj)
        for i, item in enumerate(obj):
            obj[i] = todict(item)
        if _totuple:
            obj = tuple(obj)
    elif isinstance(obj, dict):
        obj = dict(obj)
        for key, val in obj.items():
            obj[key] = todict(val)
    return obj

class ObjDict(dict):
    """ dict with object access to keys and read write to yaml files
    Examples:
    >>> d = ObjDict(**{some_key:some_value})
    >>> d.new_key = some_value      # get or set keys as objects
    >>> d.to_yaml(filename)         # write yaml
    >>> d.from_yaml(filename)       # load yaml
    >>> d.to_json(filename)         # write json
    >>> d.from_json(filename)       # load json
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
        if isinstance(value, dict):
            self[name] = ObjDict(self[name])
            self[name]._recurse_obj()

    def __delattr__(self, name: str) -> None:
        del self[name]

    def copyobj(self: _T) -> _T :
        """ .copy() returns a dict, not ObjDict"""
        return ObjDict(self.copy())

    def deepcopy(self: _T) -> _T :
        return deepcopy(self)

    def to_yaml(self, name: str) -> None:
        """ save to yaml"""
        with open(name, 'w') as _fi:
            yaml.dump(todict(self), _fi)

    def from_yaml(self,
                  name: str,
                  update: bool = False,
                  out_type: Optional[str] = None,
                  **kwargs) -> None:
        """ load yaml to dictionary
        Args
            update      (bool [False]) False overwrites, True appends
            out_type    (str [None]) numpy, torch
            valid kwargs
                dtype   (str ['float32'])
                device  (str ['cpu']) | 'cuda'
        """
        name = _get_fullname(name)
        if not update:
            self.clear()
        with open(name, 'r') as _fi:
            _dict = yaml.load(_fi, Loader=_get_yaml_loader())
            self.update(_dict)
        self._as_type(out_type, **kwargs)
        self._recurse_obj()

    def _recurse_obj(self):
        """ convert dict to ObjDict recursively
        """
        for key in self:
            if isinstance(self[key], dict):
                self[key] = ObjDict(self[key])
                self[key]._recurse_obj()

    def to_dict(self):
        """convert ObjDict to dict recurively"""
        return todict(self)

    def to_json(self, name: str) -> None:
        """save to json"""
        name = _get_fullname(name)
        with open(name, 'w') as _fi:
            json.dump(todict(self), _fi)

    def from_json(self,
                  name: str,
                  update: bool = False,
                  out_type: Optional[str] = None,
                  **kwargs) -> None:
        """load json to dictionary
        Args
            update      (bool [False]) False overwrites, True appends
            out_type    (str [None]) numpy, torch
            valid kwargs
                dtype   (str ['float32'])
                device  (str ['cuda']) | 'cpu'
        """
        if not update:
            self.clear()
        with open(name, 'r') as _fi:
            _dict = json.load(_fi)
            self.update(_dict)
        self._as_type(out_type, **kwargs)
        self._recurse_obj()

    def _as_type(self, out_type: Optional[str] = None, **kwargs) -> None:
        if out_type is not None:
            dtype = "float32" if "dtype" not in kwargs else kwargs["dtype"]
            device = "cpu" if "device" not in kwargs else kwargs["device"]
            if out_type[0] == "n":
                self.as_numpy(dtype=dtype)
            elif out_type[0] in ('p', 't'):
                self.as_torch(dtype=dtype, device=device)

    def as_numpy(self, dtype: str = "float32") -> None:
        """ converts lists and torch tensors to numpy array
            DOES not check array validity
        """
        dtype =  np.__dict__[dtype]
        for key in self:
            if isinstance(self[key], (list, tuple)):
                self[key] = np.asarray(self[key], dtype=dtype)
            elif WITH_TORCH and isinstance(self[key], torch.Tensor):
                self[key] = self[key].cpu().clone().detach().numpy()

    def as_torch(self, dtype: str = "float32", device: str = "cpu" ) -> None:
        """ converts all lists and ndarrays to torch tensor
            DOES not check array validity
            DOES not convert dimensionless data
        """
        assert WITH_TORCH, "pytorch not found, install first"
        dtype = torch.__dict__[dtype]
        for key in self:
            if isinstance(self[key], (list, tuple, np.ndarray)):
                self[key] = torch.as_tensor(self[key], dtype=dtype, device=device)

    def as_list(self) -> None:
        """ converts all tensors and ndarrays to list
        # will fail on dimensionless
        """
        for key in self:
            if isinstance(self[key], np.ndarray) or (WITH_TORCH and torch.is_tensor(self[key])):
                self[key] = self[key].tolist()

def _get_fullname(name: str) -> str:
    name = osp.expanduser(osp.abspath(name))
    os.makedirs(osp.split(name)[0], exist_ok=True)
    return name

def _get_yaml_loader(loader : Optional[str] = None) -> Any:

    loader = yaml.SafeLoader
    # parse scientific notation
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return loader
    # loaders = yaml.loader.__dict__['__all__']
    # loader = loader if loader is not None else ["FullLoader", "BaseLoader"]
    # loader = [loader] if isinstance(loader, str) else loader

    # loader = list(set(loaders) & set(loader))
    # if not loader:
    #     loader = loaders
    # return yaml.__dict__[loader[0]]


def filter_kwargs(func: Callable, kwargs: dict, complete: bool = False) -> dict:
    """ filters a set kwargs to match function args
    Args:
        func        callable
        kwargs      dict
        complete    bool [False], if True, checks that all required arguments are filled
    """
    required = []
    args = []
    for key, value in inspect.signature(func).parameters.items():
        if key == "self":
            continue
        args.append(key)
        if value.default == inspect._empty:
            required.append(key)

    out = {key:value for key, value in kwargs.items() if key in args}

    if complete:
        missing = [arg for arg in required if arg not in out]
        assert not missing, f" missing required arguments {missing}"

    return out
