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

def dirichlet_mode(x):
    return [(v - (len(x) - 1)) / (sum(x) - len(x)) for v in x]


#----------- WorkLearn-specific utils --------

def equation_safe_filename(eq):
    if isinstance(eq, basestring):
        return eq.replace('/', 'div')
    else:
        return eq

import pandas as pd
import csv

def worklearn_runtime(results_path):
    """Return experiment runtime in seconds.

    Args:
        results_path:   Path of experiment results file.

    """
    df = pd.read_csv(results_path)
    df = df[['sys_t']]
    max_time = df.max()
    min_time = df.min()
    diff = max_time - min_time
    return diff[0]

def worklearn_current_iteration(results_path):
    df = pd.read_csv(results_path)
    d = dict()
    for p, df_s in df.groupby('policy'):
        d[p] = df_s['iteration'].max()
    return d


def worklearn_res_by_ep(results_path, policyname=None):
    """Return episode runs, one by one.

    Args:
        results_path:     Path to experiment file.
    
    """
    df = pd.read_csv(results_path)
    if policyname is not None:
        df = df[df.policy == policyname]
    
    df['aor'] = zip(df['a'], df['o'], df['r'], df['b'])
    df_grouped = df.groupby(['policy','iteration','episode'])['aor'].agg(lambda x: tuple(x))

    def format_tup(t):
        try:
            a = int(t[0])
            o = int(t[1])
            r = t[2]
        except:
            a = ' '
            o = ' '
            r = 0
        b = t[3]

        return '{} ({} {:.2f}): {}'.format(
            a, o, r, 
            ' '.join('{:.2f}'.format(float(s)) for s in b.split()))
    def f():
        for i,v in df_grouped.iteritems():
            print v
            yield tuple(format_tup(t) for t in v)

    return f()
