import numpy as np

# Optimality Criteria Update Function with bi-sectioning algorithm
def OC(nelx: int, nely: int, x: np.array, volfrac: float, dc: np.array):
    l1 = 0  # lower bi-sectioning bound
    l2 = 1e5  # upper bi-sectioning bound
    move = 0.2  # sectioning increment
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l2 + l1)  # middle bi-sectioning value
        xnew = np.maximum(
            0.001,
            np.maximum(
                x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))
            ),
        )
        if np.sum(xnew) - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew