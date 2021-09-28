""" @xvdp   data utility classes
    ObjDict     object access dictionary
    TraceMem    ObjDict collecting memory usage
    GPUse       thin wrap of nvidia-smi
        TODO replace with nvml
    CPUse       thin wrap of psutils
"""
from typing import Any
import subprocess as sp
from copy import deepcopy
import os
import os.path as osp
import json
import psutil
import numpy as np
import yaml

from koreto import WITH_TORCH
if WITH_TORCH:
    import torch

__all__ = ["ObjDict", "TraceMem", "GPUse", "CPUse"]

# pylint: disable=no-member
# ###
# Dictionaries and Memory management
# 
class DeepClone:
    """ similar to deep copy detaching tensors to cpu
        self.out    cloned data
        self.stats  counter of classes cloned
    """
    def __init__(self, data, cpu=True):
        self._cpu = cpu
        self.out = None
        self.stats = {}
        self.out = self.clone(data)

    def clone(self, data):
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

    def clone_dict(self, data):
        out = data.__class__()
        for k in data:
            out[k] = self.clone(data[k])
        return out

    def clone_list(self, data):
        """ lists and tuples - should handle iterables generally"""
        return data.__class__(self.clone(d) for d in data)

def deepclone(data):
    """ similar to deep copy detaching tensors to cpu
            out = deepclone(data)
    """
    return DeepClone(data).out

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

    def __delattr__(self, name: str) -> None:
        del self[name]

    def to_yaml(self, name)-> None:
        """ save to yaml"""
        with open(name, 'w') as _fi:
            yaml.dump(dict(self), _fi)

    def from_yaml(self, name, update=False, out_type=None, **kwargs)-> None:
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

    def to_json(self, name)-> None:
        """save to json"""
        name = _get_fullname(name)
        with open(name, 'w') as _fi:
            json.dump(dict(self), _fi)

    def from_json(self, name, update=False, out_type=None, **kwargs)-> None:
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

    def _as_type(self, out_type=None, **kwargs):
        dtype = "float32" if "dtype" not in kwargs else kwargs["dtype"]
        device = "cpu" if "device" not in kwargs else kwargs["device"]
        if out_type is not None:
            if out_type[0] == "n":
                self.as_numpy(dtype=dtype)
            elif out_type[0] in ('p', 't'):
                self.as_torch(dtype=dtype, device=device)

    def as_numpy(self, dtype="float32")-> None:
        """ converts lists and torch tensors to numpy array
            DOES not check array validity
        """
        dtype =  np.__dict__[dtype]
        for key in self:
            if isinstance(self[key], (list, tuple)):
                self[key] = np.asarray(self[key], dtype=dtype)
            elif WITH_TORCH and isinstance(self[key], torch.Tensor):
                self[key] = self[key].cpu().clone().detach().numpy()

    def as_torch(self, dtype="float32", device="cpu")-> None:
        """ converts all lists and ndarrays to torch tensor
            DOES not check array validity
            DOES not convert dimensionless data
        """
        assert WITH_TORCH, "pytorch not found, install first"
        dtype = torch.__dict__[dtype]
        for key in self:
            if isinstance(self[key], (list, tuple, np.ndarray)):
                self[key] = torch.as_tensor(self[key], dtype=dtype, device=device)

    def as_list(self)-> None:
        """ converts all tensors and ndarrays to list
        # will fail on dimensionless
        """
        for key in self:
            if isinstance(self[key], np.ndarray) or WITH_TORCH and isinstance(self[key], torch.Tensor):
                self[key] = self[key].tolist()

def _get_fullname(name):
    name = osp.expanduser(osp.abspath(name))
    os.makedirs(osp.split(name)[0], exist_ok=True)
    return name

def _get_yaml_loader(loader=None):
    loaders = yaml.loader.__dict__['__all__']
    loader = loader if loader is not None else ["FullLoader", "BaseLoader"]
    loader = [loader] if isinstance(loader, str) else loader

    loader = list(set(loaders) & set(loader))
    if not loader:
        loader = loaders
    return yaml.__dict__[loader[0]]

class ObjTrace(ObjDict):
    """ dict with object access
        delta function for iterable members
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def delta(self, name, i=-1, j=-2):
        """ return self[name][i] - self[name][j]"""
        assert isinstance(self[name], (tuple, list, np.ndarray)), "cannot delta type {}".format(type(self[name]))
        assert abs(i) <= len(self[name]) and abs(j) <= len(self[name]), "indices ({} {}) outside of range {}".format(i, j, len(self[name]))
        return self[name][i] - self[name][j]

class TraceMem(ObjTrace):
    """ ObjDict to tracing memory states
    Example
    >>> m = TraceMem()
    >>> m.step(msg="") # collect /and log cpu and gpu
    >>> m.log()     # log all colected steps
    """
    def __init__(self, units="MB"):
        self.units = units
        cpu = CPUse(units=self.units)
        gpu = GPUse(units=self.units)

        self.GPU = [gpu.available]
        self.CPU = [cpu.available]

        self.dGPU = [0]
        self.dCPU = [0]

        self.msg = ["Init"]
        self.log_mem(cpu, gpu)

    def log_mem(self, cpu, gpu):
        print(f"  CPU: avail: {cpu.available} {self.units} \tused: {cpu.used} {self.units} ({cpu.percent}%)")
        print(f"  GPU: avail: {gpu.available} {self.units} \tused: {gpu.used} {self.units} ({gpu.percent}%)")

    def step(self, msg="", i=-2, j=-1, verbose=True):
        cpu = CPUse(units=self.units)
        gpu = GPUse(units=self.units)
        self.CPU += [cpu.available]
        self.GPU += [gpu.available]
        self.msg +=[msg]
        dCPU = self.delta('CPU', i=i, j=j)
        dGPU = self.delta('GPU', i=i, j=j)
        self.dGPU += [dGPU]
        self.dCPU += [dCPU]

        if verbose:
            msg = msg + ": " if msg else ""
            print(f"{msg}Used CPU {dCPU}, GPU {dGPU} {self.units}")
            self.log_mem(cpu, gpu)

    def log(self):
        print("{:^6}{:>12}{:>12}{:>12}{:>12}".format("step", "CPU avail", "CPU added",
                                                     "GPU avail", "GPU added"))
        for i in range(len(self.GPU)):
            print("{:^6}{:>12}{:>12}{:>12}{:>12}  {:<6}".format(i, f"{self.CPU[i]} {self.units}",
                                                                f"({self.dCPU[i]})",
                                                                f"{self.GPU[i]} {self.units}",
                                                                f"({self.dGPU[i]})", self.msg[i]))

def get_smi(query):
    _cmd = ['nvidia-smi', '--query-gpu=memory.%s'%query, '--format=csv,nounits,noheader']
    return int(sp.check_output(_cmd, encoding='utf-8').split('\n')[0])

class GPUse:
    """thin wrap to nvidia-smi"""
    def __init__(self, units="MB"):
        self.total = get_smi("total")
        self.used = get_smi("used")
        self.available = self.total - self.used
        self.percent = round(100*self.used/self.total, 1)
        self.units = units if units[0].upper() in ('G', 'M') else 'MB'
        self._fix_units()

    def _fix_units(self):
        if self.units[0].upper() == "G":
            self.units = "GB"
            self.total //= 2**10
            self.used //= 2**10
            self.available //= 2**10

    def __repr__(self):
        return "GPU: ({})".format(self.__dict__)

class CPUse:
    """thin wrap to psutil.virtual_memory to matching nvidia-smi syntax"""
    def __init__(self, units="MB"):
        cpu = psutil.virtual_memory()
        self.total = cpu.total
        self.used = cpu.used
        self.available= cpu.available
        self.percent = cpu.percent
        self.units = units if units[0].upper() in ('G', 'M') else 'MB'
        self._fix_units()

    def _fix_units(self):
        _scale = 20
        if self.units[0].upper() == "G":
            self.units = "GB"
            _scale = 30
        else:
            self.units = "MB"
        self.total //= 2**_scale
        self.used //= 2**_scale
        self.available //= 2**_scale

    def __repr__(self):
        return "CPU: ({})".format(self.__dict__)
