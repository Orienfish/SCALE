import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
import ctypes
from functools import partial


def l2_dist(x, y):
    
    xx = np.sum(x**2, axis = 1)
    yy = np.sum(y**2, axis = 1)
    xy = np.dot(x, y.transpose((1,0)))

    d = xx[:,None] - 2*xy + yy[None,:]
    d[d<0] = 0
    d = (d)**(1/2)
    
    return d

def l1_dist(x, y):
    return np.sum(np.absolute(x[:,None,:]-y[None,:,:]), axis = 2)


def earth_mover_dist(x, y):
    return pairwise_distances(x, y, metric=wasserstein_distance)

def smart_dist(x, y, method="L2"):
    if method == "L1":
        return l1_dist(x, y)
    elif method == "L2":
        return l2_dist(x, y)
    elif method == "EM":
        return earth_mover_dist(x, y)
    else:
        return x
