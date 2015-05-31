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


#----------- WorkLearn-specific utils --------

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

def make_paths(results_path, policyname=None):
    """Pretty print paths.

    Args:
        results_path:     Path to experiment file.
    
    Assumes sorted by iteration, timestep.
    """
    # TODO: Handle any order of rows.
    fp = open(results_path, 'r')
    reader = csv.DictReader(fp)

    runs = []
    for row in reader:
        if policyname is not None and row['policy'] != policyname:
            continue

        if int(row['iteration']) == 0 and int(row['t']) == 0:
            runs.append([])
        if row['a'] != '':
            runs[-1].append('{} ({}, {})'.format(row['a'], row['o'], row['r']))
    for run in runs:
        print '----'
        print ' '.join(run)

    fp.close()
