"""@xvdp
memory tools

current snapshots
GPUse() -> current cuda use # subprocess wrapper over
CPUse() -> current cpu use

decorator using nvml torch.cuda.memory_stat and torch.profiler.profile
@memory_profiler
"""

from os import stat
import subprocess as sp
import psutil
import numpy as np
import torch
import pynvml as nvml  # pip install nvidia-ml-py
from torch.profiler import profile, ProfilerActivity

from koreto.camera import pixels_to_rays
from .utils import ObjDict


def has_cuda(*args, **kwargs):
    """ recursive check if theres a cuda object in the inputs
    """
    next_args = []
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor) and arg.device.type == "cuda":
            return True
        elif isinstance(arg, (list, tuple, set)):
            next_args += list(arg)
    if next_args:
        return has_cuda(*next_args)
    return False


# pylint: disable=no-member
class memory_profiler:
    """ profile memory usage and times
    digest of:
        free memory diff before and after function from `nvml`
        peak memory use from `torch.cuda.memory_stats()`
        cpu, cuda memory and time for costlier operations from `torch.profiler`

    Example:
    >>> @memory_profiler
    >>> def a():
    >>>     t = torch.randn([1,3,3000,300])
    >>>     t.div_(45.)
    >>>     return t
    >>> x = a()
    """
    def __init__(self, func):
        self.func = func

    def _call_no_cuda(self, *args, **kwargs):
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            out = self.func(*args, **kwargs)
        _name = f"{self.func.__name__}(): profile memory"
        print(f"\n{_name:35} {'CPU':8}")
        timed_events = []
        for event in prof.key_averages():
            _time =  event.cpu_time_total
            if _time:
                timed_events.append((self.format_event_name(event.__dict__['key'], 35), _time, event.cpu_time_total))
            cpu = event.cpu_memory_usage >> 20
            if cpu:
                cpu = f"{cpu} MB" if cpu else "  -  "
                print(f"  {self.format_event_name(event.__dict__['key'], 35):35} {cpu:8}")

        _name = f"{self.func.__name__}(): profile time"
        print(f"\n{_name:35} {'CPU':8}")
        timed_events = sorted(timed_events, key=lambda x: x[1], reverse=True)
        for event in timed_events[:10]:
            print(f"  {event[0]:35} {self.msus(event[2]):8}")

        return out

    def __call__(self, *args, **kwargs):
    
        if not has_cuda(*args, **kwargs) and not torch.cuda.is_initialized():
            return self._call_no_cuda(*args, **kwargs)

        # clear cuda cached memory and peak
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # print availabe cuda before running function
        print(f"@memory_profiler\nnvml snapshot before '{self.func.__name__}()':")
        nvml.nvmlInit()
        device_indices = list(range(nvml.nvmlDeviceGetCount()))
        infos = [nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(i)) for i in device_indices]

        free = []
        for i, device in enumerate(infos):
            print(f"  cuda:{i}\ttotal {device.total >> 20} MB, used {device.used >> 20} MB, free {device.free >> 20} MB")
            free.append(device.free)

        self.cuda_torch_peak(f"before {self.func.__name__}()")

        # function call
        # collect per call times and use
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            out = self.func(*args, **kwargs)

        # trace memory by step
        _name = f"{self.func.__name__}(): profile memory"
        print(f"\n{_name:35} {'CPU':8}\t {'CUDA':8}")
        timed_events = []
        for event in prof.key_averages():
            _time =  max(event.cuda_time_total, event.cpu_time_total)
            if _time:
                timed_events.append((self.format_event_name(event.__dict__['key'], 35),
                                     _time, event.cpu_time_total, event.cuda_time_total))
            cpu = event.cpu_memory_usage >> 20
            cuda = event.cuda_memory_usage >> 20
            if cpu or cuda:
                cpu = f"{cpu} MB" if cpu else "  -  "
                cuda = f"{cuda} MB" if cuda else "  -  "
                print(f"  {self.format_event_name(event.__dict__['key'], 35):35} {cpu:8}\t {cuda:8}")

        # sort by time used
        _name = f"{self.func.__name__}(): profile time"
        print(f"\n{_name:35} {'CPU':8}\t {'CUDA':8}")
        timed_events = sorted(timed_events, key=lambda x: x[1], reverse=True)
        for event in timed_events[:10]:
            print(f"  {event[0]:35} {self.msus(event[2]):8}\t {self.msus(event[3]):8}")

        currents, peaks = self.cuda_torch_peak("after indices")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        for i in device_indices:
            current2 = 0
            for stat, val in torch.cuda.memory_stats(device=i).items():
                if 'current' in stat and val > current2:
                    current2 = val

            if current2 < currents[i]:
                print(f"\nunreleased memory: {currents[i] >> 20:8} MB -> {current2 >> 20} MB")

        # measure nvml after function exits
        used = [nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(i))
                for i in device_indices]
        print(f"nvml dif:")
        for i, device in enumerate(used):
            print(f"  cuda:{i}\tfree: {free[i] >> 20} MB -> {device.free >> 20} MB ({(free[i] - device.free) >> 20})")

        nvml.nvmlShutdown()

        return out

    @staticmethod
    def cuda_torch_peak(msg=""):
        currents = []
        peaks =[]
        device_indices = list(range(torch.cuda.device_count()))
        print(f"\ncuda.memory_stats() {msg}")
        for i in device_indices:
            peak = 0
            current = 0
            for stat, val in torch.cuda.memory_stats(device=i).items():
                if val > peak:
                    peak = val
                if 'current' in stat and val > current:
                    current = val
            print(f"  cuda:{i}\tpeak: {peak >> 20:8} MB\t current {current >> 20:8} MB")
            peaks.append(peak)
            currents.append(current)
        return currents, peaks

    @staticmethod
    def msus(x):
        if x > 0:
            return f"{int(x)} us" if x < 1000 else f"{int(x//1000)} ms"
        return  "  -  "
    @staticmethod
    def format_event_name(x, chars=35):
        name = f"{x}"
        if len(name) > chars:
            name = name[:chars -1]+'~'
        return name

class ObjTrace(ObjDict):
    """ dict with object access
        delta function for iterable members
    """
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
