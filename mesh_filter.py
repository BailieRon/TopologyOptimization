import numpy as np

def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))  # initialize
    rmin_floor = int(np.floor(rmin))

    for i in range(nelx):  # first element in dependency check
        for j in range(nely):  # second element in dependency check
            sum = 0.0
            for k in range(max(i - rmin_floor, 0), min(i + rmin_floor + 1, nelx)):
                for l in range(max(j - rmin_floor, 0), min(j + rmin_floor + 1, nely)):
                    fac = rmin - np.sqrt(
                        (i - k) ** 2 + (j - l) ** 2
                    )  # weighting factor with rmin as filter size minus distance between two elements
                    if fac > 0:
                        sum += fac
                        dcn[j, i] += fac * x[l, k] * dc[l, k]
            if sum > 0:
                dcn[j, i] /= x[j, i] * sum
    return dcn