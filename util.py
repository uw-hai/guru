"""util.py

General utilities

"""

import os

def get_or_default(d, k, default_v):
    try:
        return d[k]
    except KeyError:
        return default_v

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 

#--------- Statistics ------

from scipy.special import gamma
import scipy.stats as ss

beta = ss.beta.pdf

def dbeta(x, a, b):
    return gamma(a+b)/(gamma(a)*gamma(b)) * \
           ((a-1) * x**(a-2) * (1-x)**(b-1) - x**(a-1) * (b-1) * (1-x)**(b-2))

assert dbeta(0.5, 2, 2) == 0
assert dbeta(0.6, 2, 2) != 0
