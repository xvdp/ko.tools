"""@
distribution and bound sketches, from notebooks/fr/Concentration.ipynb

n choose k
    using log sums
    using log gamma
bernoulli

factorial
bounds
    chebyshev
    chernoff
    talangrand

"""
from typing import Union
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt




def chernoff_bound(number, sym=False):
    """P(|X-m| >= ε*μ) <= 2exp(-(ε^2μ)/(2+ε))
    t >= 0 and flip, otherwise it is offset

    **[Chernoff bound] https://courses.cs.washington.edu/courses/cse312/18wi/312A/lecture21.pdf**
    **https://courses.cs.washington.edu/courses/cse312/21wi/files/slides/Lecture-23-Chernoff-Bounds.pdf**
    bound is an exponentially decreasing upper bound on the tail of a random variable.
    Deviation of independent random vars from expectations

    if $0 < \delta$ then
        $$p(X \geq (1+\delta)\mu) \leq exp(-\frac{\delta^2 \mu}{2+\delta})$$
        and
        $$ p(X \leq (1-\delta)\mu) \leq exp(-\frac{\delta^2 \mu}{2})$$
        thus
    $$p(|X -\mu| > \delta\mu) \leq 2exp(-\frac{\delta^2 \mu}{2+\delta})$$
    """
    mean = number/2
    even=0
    if not number%2 and sym:
        even = 1
    x = torch.arange(0, int(mean)+even )
    eps = x/mean
    y = 2 * torch.exp(-(eps**2*mean)/(2+eps) )
    if sym:
        x = torch.cat(((x - len(x))[1:], x[:-1]))
        y = torch.cat((y[1:].flip(-1), y[:-1]))
    return x, y

def chebyshev_bound(number, var=None, p=0.5, sym=False):
    """
    Chebyshev
    $$ \mathbb{P}(|X - \mathbb{E}(X)|\geq t) \leq \frac{Var(X)}{t^2}$$ 
    Binomial Var(X) = np(1-p)
    """
    mean = number/2
    var = var if var is not None else lambda number, p: number*p*(1-p)
    x = torch.arange(-int(mean) if sym else 0, int(mean))
    y = var(number, p)/x**2
    return x,y


def talagrand_bound(number, sym=False, op=torch):
    """
    Tallagrand 1996,
    [A Look At Indepenence](https://www.cmat.edu.uy/~lessa/tesis/Talagrand%20-%20newlook.pdf)
    Consider an independent seqience of Bernoulli random variables $(\epsilon_i)_{i \leq N}$ 
    i.e. $ P(\epsilon_i=1= (\epsilon_i=-1)=1/2$ . Then for all  $ t \geq 0$ (proven in (4.7))
    \begin{equation} \tag{1.2}
    P (| \sum_{i \leq N} \epsilon_i | \geq t ) \leq 2 exp\left( -\frac{t^2}{2N}\right)
    \end{equation}
    If $B_N$ == number of ones in $(\epsilon_i)_{i\leq N}$ : then
    $ \sum_{i \leq N} \epsilon_i = 2B_N - N$,  so (1.2) $\equiv$
    """
    mean = number/2
    x = op.arange(-int(mean) if sym else 0, int(mean))
    y = 2* op.exp((-2*x**2)/number)  #eval_P_at_t_N(x,N)
    return x, y


def log_n_choose_k(n):
    """return all log (n k)"""
    _n = torch.cat((torch.ones(1), torch.arange(1, n+1))).log().cumsum(axis=0)
    return _n[-1] - (_n + _n.flip(-1))


def log_n_choose_k_gamma(n, k):
    """ n choose k using log gamma distribution"""
    n = torch.as_tensor(n)
    k = torch.as_tensor(k)
    return ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma())


def bernoulli(n, p=0.5):
    """ condensed version of bernouilli distribution
    """
    k = torch.arange(0, n+1)
    _n = torch.where(k > 0, k , 1).log().cumsum(axis=0)
    return p**k * (1-p)**(n-k) * (_n[-1] - (_n + _n.flip(-1))).exp()


# verbose factorials

def factorial(n: int, fmt: Union[int, float, None] = None)-> Union[int, float, str]:
    """ 
    Args:
        n   (int) n <= 171: float64; n <= 1754: float128; n > 1754:log10(factorial)
        fmt (int, str, float [None]) format
            None            returns max dtype permitted by 'n'
            1, 'str'        returns str
            np.e, 'log'     log(factorial)
            2, 'log2'       log2(factorial)
            10, 'log10'     log10(factorial)
            float_any       log(factorial)/log(float_any)

    Args
        fmt (str) 'log2' | 'log10' | 'str'
            (int) 1: 'str' > 1: log(out)/log(fmt)
        np.finfo(np.float32).max := 3.4028235e+38
        np.finfo(np.float64).max := 1.7976931348623157e+308
        np.finfo(np.float128).max := 1.189731495357231765e+4932
    """
    _mul = -2*(n%2) + 1 if n < 1 else  1

    _asstr = lambda x, n: x if n > 20 else f"{x:,}"
    n = np.abs(n)
    if n < 3:
        out = n
    else:
        seq = np.arange(n,1,-1)
        if isinstance(fmt, str) and fmt[0] == 'l':
            fmt = {'log':np.e, 'log2':2, 'log10':10}[fmt]

        if (isinstance(fmt, (int, float)) and fmt > 1) or n > 1754:
            if n > 1754 and fmt is None:
                warnings.warn(f"{n}! exceedes float128 max 1.189e+4932, returning log10")
                fmt = 10
            den = 1 if fmt == np.e else np.log(fmt)
            out = np.sum(np.log(seq)/den)
        else:
            dtype = np.float64 if n < 171 else np.longdouble
            out = np.prod(seq.astype(dtype))
            # out = np.prod(np.arange(n,1,-1, dtype=dtype))
            if out <= np.iinfo(np.int64).max:
                out = int(out)
    out *= _mul

    if (isinstance(fmt, int) and fmt == 1) or (isinstance(fmt, str) and fmt[0] == 's'):
        out = _asstr(out, n)
    return out


def log_factorial(n: Union[int, float, np.ndarray, torch.Tensor],
                  op = torch,
                  reduction='cumsum'):
    """ return sum or cumsum, log n or log n, n-1, ..., 1
    Args
        n   (int, float) n, n-1, ..., 1
            (ndarray, tensor)
        op  (module) torch, np
        reduction (str) 'sum'       ->  log(n!)
                        'cumsum'    ->  log(n!), log((n-1)!), ..., log(1!)
    max n if exponentiated
    float32 n in {3, 33}, out <= 88.5808
    float64 n in {34, 170}, out <= 706.5731
    float128 in in {171, 1754} out <= 11352.4277
    """ # op.where(n > 0, n, 1)
    if isinstance(n, np.ndarray) and len(n) > 1:
        op = np
    elif torch.is_tensor(n) and len(n) > 1:
        op = torch
    else:
        n = op.arange(1, n + 1)
    if reduction == 'sum':
        out = op.sum(op.log(n), axis=0)
    else:
        out = op.cumsum(op.log(n), axis=0)
    return out



def BernoulliPMF(n=50, p=0.5, t: Union[int, tuple] = None, op=np):
    """ n=50 is 51 tosses, not 50
    Args
        n: (int) number of tosses
        p: (float [0.5]) 0 < p < 1, probability
        t: (int) divergence from center, number of extra heads
        t: (tuple) probability that distribution falls in range;
            eg, n=50, t=(0, None) : P(t>=0)
            eg, n=50, t=(-3,3) : P(-3<=t<=3) p that tosses fall between -3 and 3 off center for a run of 50
            eg, n=50, t=(0,10) : P(0<=t<=10)
    $$ B(k;n;p) = p^k(1-p)^{n-k} \frac{n!}{k!(n-k)!} $$
    $p^k(1-p)^{n-k}  exp (\sum_{i\leq N}ln(i) - \sum_{i\leq k}\sum_{j\leq i}  ln(j) - \sum_{i\leq n-k}\sum_{j\leq i}  ln(j))$
    """
    k = op.arange(0, n+1)
    kp = op.cumsum(op.log(op.where(k>0,k,1)), axis=0)
    kp = kp + kp.flip(-1) if op == torch else kp + kp[::-1]

    out =  p**k * (1-p)**(n-k) * op.exp(op.sum(op.log(op.where(k>0,k,1))) - kp)
    if t is not None:
        if isinstance(t, (tuple, list)) and t[1] is not None and t[1]- t[0] == 1:
            t = t[0]

        if isinstance(t, int):
            return out[slice(t+int(np.floor((n+1)/2)) , t+1+int(np.ceil(n/2)))].mean()

        elif isinstance(t, (tuple, list)):
            _halfcenter = 0
            start, stop = t
            if not op.fmod(n,2):
                _halfcenter = out[max(0, start + n//2)]/2
                if stop is not None and stop + n//2 < n + 1:
                    _halfcenter += out[stop + n//2]/2
                start += 1
            else:
                start += 1
                stop = stop if stop is None else stop + 1

            start = max(start + n//2, 0)
            stop = stop if stop is None else stop + n//2
            return out[slice(start, stop)].sum() + _halfcenter

    return out


def toss(N=100, runs=10000, device="cpu"):
    # toss 'runs' number of 'N' tosses
    return torch.bernoulli(torch.full((runs, N), 0.5, device=device))*2 - 1

def BN_minus_halfN(tosses, N):
    ones = (tosses == 1).sum(dim=1)
    return torch.abs(ones - N/2)

def sum_tosses(tosses):
    return tosses.sum(dim=1).abs()

def plot_hist(x, title=None, figsize=None, subplot=None, show=True,
              xlabel=None, text=None, grid=True, label=None,
              N=None, xticks=None, yticks=None, suptitle=None):
    if figsize:
        plt.figure(figsize=figsize)
    if suptitle:
        plt.suptitle(suptitle)
    if subplot:
        plt.subplot(subplot)
    if title:
        plt.title(title)
    max_size = int(x.max().item())
    min_size = int(x.min().item())
    dh = torch.histc(x.cpu(), bins=max_size- min_size)
    perc_outcomes = dh/len(x)

    step = (max_size-min_size)//5
    if min_size < 0:
        __xticks = torch.cat((torch.arange(0-step, min_size,-step).flip(-1),
                              torch.arange(0,max_size+step, step)))
    else:
        __xticks = torch.arange(min_size, max_size+step, step)

    if N:
        head = []
        tail = []
        if N/2 + min_size > 0:
            head = torch.zeros(N//2 + min_size)
        if N/2 > max_size:
            tail = torch.zeros(N//2 - max_size)
        perc_outcomes = torch.cat((head, perc_outcomes, tail))

        _xticks = [0] + (__xticks +N//2).tolist() + [N]
        __xticks = _xticks
    else:
        _xticks = (__xticks - min_size).tolist()
        __xticks = __xticks.tolist()

    if xticks is not None:
        _xticks = __xticks = xticks

    plt.plot(perc_outcomes, label=label)
    plt.ylabel("$\mu$%")
    if grid:
        plt.grid()
    plt.xticks(_xticks,  xticks)
    if yticks is not None:
        plt.yticks(yticks)

    if xlabel:
        plt.xlabel(xlabel)
    # d = dict(zip(range(min_size, max_size), dh.to(torch.int).tolist()))
    # print(f"value: number\n{d}")
    if text is not None:
        plt.text(*text)
    if show:
        if label is not None:
            plt.legend()
        plt.tight_layout()
        plt.show()
    return perc_outcomes.max()

def int_tensor_str(x):
    return str(x.to(dtype=torch.int64, device='cpu')).replace('tensor', '')

def plot_bounds(N = 100, runs= 10000):
    mean = N/2
    distint = ((toss(N, runs, 'cuda') == 1).sum(dim=1)-  mean)
    x,y = chebyshev_bound(N, sym=True)
    xticks=[0,mean,N] + [int(distint.min().item()+mean), int(distint.max()+mean)]
    _chop_top = plot_hist(distint, N=N, figsize=(12,4), subplot=121,
                        title=f"Binomial dist N={N}", show=False, 
                        label="Binomial histogram |$\sum  B_n - \\frac{n}{2}$|, 1M runs",
                        xticks=xticks)

    plt.ylim(0, _chop_top*1.5)
    plt.plot(BernoulliPMF(N), linestyle=":",
             linewidth=3, label="Binomial PMF $\\binom{n}{k}p^kq^{n-k}  $")
    #chebysev
    plt.plot(x+mean, y, linestyle="--",
             label="Chebyshev bound: $\leq \\frac{Var(X)}{t^2} = \\frac{np(1-p)}{t^2} $")

    x,y = chernoff_bound(N, sym=True)
    plt.plot(x+mean, y, linestyle="--",
             label="Chernoff bound: $\leq 2 exp \\left( \\frac{-t^2 \mu}{2+t} \\right)$")

    x,y = talagrand_bound(N, sym=True)
    plt.plot(x+mean, y, linestyle="--",
             label="Talagrand bound: $\leq 2 exp \\left( \\frac{-2t^2}{N} \\right)$")
    plt.legend()

    N = 800
    mean = N/2
    distint = ((toss(N, runs, 'cuda') == 1).sum(dim=1)-  mean)

    xticks=[0,mean,N] + [int(distint.min().item()+mean), int(distint.max()+mean)]
    _chop_top = plot_hist(distint, N=N, subplot=122,
                        title=f"Binomial dist N={N}", show=False,
                        label="Binomial histogram |$\sum B_n - \\frac{n}{2}$|, 1M runs",
                        xticks=xticks)
    plt.ylim(0, _chop_top*1.5)
    plt.plot(BernoulliPMF(N), linestyle=":", linewidth=3,
             label="Binomial PMF $ \\binom{n}{k} p^kq^{n-k}$")
    x,y = chebyshev_bound(N, sym=True)
    plt.plot(x+mean, y, linestyle="--",
             label="Chebyshev bound: $\leq \\frac{Var(X)}{t^2} = \\frac{np(1-p)}{t^2} $")
    x,y = chernoff_bound(N, sym=True)
    plt.plot(x+mean, y, linestyle="--",
             label="Chernoff bound: $\leq 2 exp \\left( \\frac{-t^2 \mu}{2+t} \\right)$")
    x,y = talagrand_bound(N, sym=True)
    plt.plot(x+mean, y, linestyle="--",
             label="Talagrand bound: $\leq 2 exp \\left( \\frac{-2t^2}{N} \\right)$")
    plt.legend()
    plt.tight_layout()
    plt.show()
