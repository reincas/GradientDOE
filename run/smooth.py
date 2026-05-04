from typing import cast, Any

import numpy as np

def soft_clip(x, xmax, xfuzz):
    """ Clip the values of x to the range 0..xmax smoothly using a circular arc transition with radius xfuzz."""

    assert xmax > 0
    assert xfuzz > 0
    assert xmax > 2 * xfuzz

    def soft_edge(x, xfuzz):
        result = np.copy(x).astype(float)
        mask = (x >= 0) & (x < xfuzz)
        xm = x[mask]
        result[mask] = np.sqrt(2 * xm * xfuzz - xm ** 2)
        return result

    x = xmax - soft_edge(xmax - x, xfuzz)
    x = soft_edge(x, xfuzz)
    return x

np.set_printoptions(formatter=cast(Any, {'float': '{: .3f}'.format}), linewidth=120)
x = np.array([-1, 0, 0.02, 0.1, 0.5, 0.9, 1.2, 8.9, 9.1, 9.5, 9.9, 9.98, 10.1])
print(x)
print(soft_clip(x, 10, 1))
