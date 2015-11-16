"""util.py"""

from research_utils.util import *
import pandas as pd

def get_penalty(accuracy, reward=1):
    """Return penalty needed for this accuracy to have expected reward 0.

    >>> round(get_penalty(0.9), 10)
    -9.0
    >>> round(get_penalty(0.75), 10)
    -3.0
    >>> round(get_penalty(0.5), 10)
    -1.0

    """
    return accuracy * reward / (accuracy - 1)

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
    """Find max iterations done.

    Returns: Dataframe from policy to number max iteration done, or None.

    """
    try:
        df = pd.read_csv(results_path)
    except ValueError:
        return None
    d = dict()
    for p, df_s in df.groupby('policy')['iteration']:
        d[p] = df_s.nunique()
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
