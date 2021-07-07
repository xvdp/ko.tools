""" @xvdp   data utility classes
    ObjDict     object access dictionary
    TraceMem    ObjDict collecting memory usage
    GPUse       thin wrap of nvidia-smi
    CPUse       thin wrap of psutils
"""
from typing import Any
import subprocess as sp
import os
import os.path as osp
import psutil

import numpy as np
import yaml

__all__ = ["ObjDict", "TraceMem", "GPUse", "CPUse"]
# pylint: disable=no-member

# ###
# Memory management
#
class ObjDict(dict):
    """ dict with object access to keys and read write to yaml files
    Examples:
    >>> d = ObjDict(**{some_key:some_value})
    >>> d.new_key = some_value      # get or set keys as objects
    >>> d.to_yaml(filename)         # write yaml
    >>> d.from_yaml(filename)       # load yaml
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

    def to_yaml(self, name):
        """ save to yaml"""
        os.makedirs(osp.split(name)[0], exist_ok=True)
        with open(name, 'w') as _fi:
            yaml.dump(dict(self), _fi)

    def from_yaml(self, name):
        """ load yaml"""
        with open(name, 'r') as _fi:
            _dict = yaml.load(_fi, Loader=get_loader())
            self.update(_dict)

def get_loader(loader=None):
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
        print("{:^6}{:>12}{:>12}{:>12}{:>12}".format("step", "CPU avail", "CPU added", "GPU avail", "GPU added"))
        for i in range(len(self.GPU)):
         print("{:^6}{:>12}{:>12}{:>12}{:>12}  {:<6}".format(i, f"{self.CPU[i]} {self.units}", f"({self.dCPU[i]})", f"{self.GPU[i]} {self.units}", f"({self.dGPU[i]})", self.msg[i]))


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
