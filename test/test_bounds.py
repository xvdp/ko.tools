
import numpy as np
from koreto.bounds import factorial

def test_factorial():
    assert factorial(20, np.e) == factorial(20, 'log')
    assert np.allclose(factorial(20, 'log'), np.log(factorial(20)))
    assert factorial(20, 10) == factorial(20, 'log10')
    assert np.allclose(factorial(20, 10), np.log10(factorial(20)))
    assert factorial(20, 'str') == factorial(20,1)
    assert isinstance(factorial(17), int)
    assert factorial(171).dtype == factorial(1754).dtype == np.longdouble
    assert factorial(170).dtype == np.float64
    assert factorial(1755) is not None # this should work and return log10 of factorial
    assert factorial(17540, 10) is not None
    print("OK")
    