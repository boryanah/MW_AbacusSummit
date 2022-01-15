import numba as nb
import numpy as np

@nb.njit
def dist(pos1, pos2, L=None):
    '''
    Calculate L2 norm distance between a set of points
    and either a reference point or another set of points.
    Optionally includes periodicity.
    
    Parameters
    ----------
    pos1: ndarray of shape (N,m)
        A set of points
    pos2: ndarray of shape (N,m) or (m,) or (1,m)
        A single point or set of points
    L: float, optional
        The box size. Will do a periodic wrap if given.
    
    Returns
    -------
    dist: ndarray of shape (N,)
        The distances between pos1 and pos2
    '''
    
    # read dimension of data
    N, nd = pos1.shape
    
    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1
    
    dist = np.empty(N, dtype=pos1.dtype)
    
    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist
