"""util.py"""

from research_utils.util import *
import pandas as pd

def equation_safe_filename(eq):
    if isinstance(eq, basestring):
        return eq.replace('/', 'div').replace('math.', '')
    else:
        return eq

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
