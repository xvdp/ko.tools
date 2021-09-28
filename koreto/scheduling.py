"""
"""
import logging
import numpy as np

class Schedule:
    """ learning rate scheduler
    * linear
    * exponential
    * noisy periodic functions

    Args
        start  (float)
        end    (float)
        steps  (int)
        exponential (bool [True])
            equivalent to
            >>> lrs = np.logspace(np.log10(start), np.log10(end), steps)
            >>> for i in range(steps):
            >>>    lr = lrs[i]

    Examples:
    >>> sched = Schedule(start, end, steps, exponential=True)
    # adds a cosine perturbation decaying to zero, with magnitude of end lr
    >>> sched.add_perturbation(period=steps//10, magnitude=end, fun=np.cos, decay=1)
    >>> for i in range(steps):
    >>>    lr = sched.step(i)
    """
    def __init__(self, start, end, steps, exponential=True):
        self.exp = exponential
        self.steps = steps
        self.start = np.log10(start) if exponential else start
        self.end = np.log10(end) if exponential else end
        self.interval = (self.end - self.start)/(self.steps-1)
        self.perturb = []

    def add_perturbation(self, period, magnitude, fun=np.sin, period_noise=0, mag_noise=0, decay=0):
        """
        Args
            period     int, fraction of steps
            magnitude  float, fraction of values
            fun        periodic fucntion, [np.sin]
            period_noise  float 0-1 [0: nonoise]random shift over the period
            mag_noise     float > 0 [0: no noise]

        """
        shift = lambda i: i
        if period_noise:
            i = lambda i, period_noise, period: i + int(period_noise*np.random.randint(period_noise))
            shift = lambda i: i + int(period_noise * np.random.randint(-period//2,period//2))

        scale = lambda y: y
        if mag_noise:
            scale = lambda y: y*(1 + np.random.random(1) * mag_noise)

        self.perturb.append(lambda val, i: val +  (fun(np.pi* shift(i)/period) * scale(magnitude))*(1 - i*decay/self.steps))

    def step(self, i):
        if i >= self.steps:
            logging.warning(f"ExpSchedule step {i} > last step {self.steps -1}")
        out = self.start + self.interval * i
        if self.exp:
            out = 10**out

        for perturb in self.perturb:
            out = perturb(out, i)
        return out
