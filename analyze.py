"""analyze.py

Analyze work_learn output.

"""

import time
import os
import csv
import argparse
import collections
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns

def str_action(a):
    reserved_actions = ['EXP', 'NOEXP', 'ASK', 'BOOT']
    N_RES_A = len(reserved_actions)
    a = int(a)
    if a < N_RES_A:
        s = reserved_actions[a]
    else:
        s = "Q{}".format(a - N_RES_A)
    return '{} ({})'.format(a, s)

def str_state(d=None):
    if d == None:
        return lambda s: s
    return lambda s: '{} ({})'.format(s, d['state', s])


def str_observation(o):
    o_names = ['WRONG', 'RIGHT', 'TERM']
    o = int(o)
    return '{} ({})'.format(o, o_names[o])

def step_by_t(df, interval=0.1):
    dfs = []
    for i in np.arange(0, 1, interval):
        df2 = df.copy()
        df2['t'] = df2['t'].map(lambda x: x + i)
        dfs.append(df2)
    df_out = pd.concat(dfs, ignore_index=True)
    df_out.sort(['policy','t'], inplace=True)
    df_out.reset_index()
    return df_out

def plot_cumrewards(infile, outdir):
    dr = csv.DictReader(infile)
    data = dict()
    for r in dr:
        p = r['policy']
        t = int(r['t'])
        it = int(r['iteration'])
        if r['r'] == '':
            r = 0.0
        else:
            r = float(r['r'])

        data[p,it,t] = r

    max_t = max(k[2] for k in data)
    max_it = max(k[1] for k in data)

    rewards = dict()
    cumrewards = dict()
    policies = set(k[0] for k in data)
    for p in policies:
        rewards[p] = np.zeros((max_it+1, max_t+1))

    for p,it,t in data:
        rewards[p][it,t] = data[p,it,t]

    for p in policies:
        cumrewards[p] = np.cumsum(rewards[p], 1)

    # Plot
    x = pd.Series(range(max_t+1), name='t')
    conditions = pd.Series(sorted(policies), name='policy')
    v = np.dstack([cumrewards[k] for k in sorted(policies)])

    plt.figure()
    sns.tsplot(v, time=x, condition=conditions, value='cumulative reward')
    plt.savefig(os.path.join(outdir, 'r_cum.png'))
    plt.close()

    infile.seek(0)

def plot_beliefs(df, outdir, t_frac=1.0, formatter=None):
    df = df[df.t <= df.t.max() * t_frac]

    df_b = pd.DataFrame(df.b.str.split().tolist()).astype(float)
    states = df_b.columns

    df = df.join(df_b)
    b_sums = df.groupby(['policy','t'], as_index=False).sum()
    if formatter:
        b_sums.rename(columns=dict((x, formatter(x)) for x in states), inplace=True)
        states = [formatter(x) for x in df_b.columns]
    for p, df_b in b_sums.groupby('policy', as_index=False):
        step_by_t(df_b).plot(x='t', y=states, kind='area',
                             title='Belief counts', logx=True)
        fname = os.path.join(outdir, 'b_{}'.format(p))
        plt.savefig(fname + '.png')
        plt.close()
        df_b.to_csv(fname + '.csv')

def plot_actions(df, outdir, t_frac=1.0, formatter=None):
    df = df[df.t <= df.t.max() * t_frac]

    actions = df.groupby(['policy', 't'])['a'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        actions.rename(columns=dict((x, formatter(x)) for x in actions.columns[2:]), inplace=True)
    for p, df_a in actions.groupby('policy', as_index=False):
        step_by_t(df_a).plot(x='t', y=actions.columns[2:], kind='area',
                             title='Action counts', logx=True)

        fname = os.path.join(outdir, 'a_{}'.format(p))
        plt.savefig(fname + '.png')
        plt.close()
        df_a.to_csv(fname + '.csv')

def plot_observations(df, outdir, t_frac=1.0, formatter=None):
    df = df[df.t <= df.t.max() * t_frac]

    obs = df.groupby(['policy', 't'])['o'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        obs.rename(columns=dict((x, formatter(x)) for x in obs.columns[2:]), inplace=True)
    for p, df_o in obs.groupby('policy', as_index=False):
        step_by_t(df_o).plot(x='t', y=obs.columns[2:], kind='area',
                             title='Observation counts', logx=True)
        fname = os.path.join(outdir, 'o_{}'.format(p))
        plt.savefig(fname + '.png')
        plt.close()
        df_o.to_csv(fname + '.csv')

def parse_names(f):
    d = dict()
    dr = csv.DictReader(f)
    for r in dr:
        d[r['type'],int(r['i'])] = r['s']
    return d


if __name__ == '__main__':
    start_time = time.clock()

    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'))
    parser.add_argument('-o', '--outdir', type=str)
    parser.add_argument('-n', '--names', type=argparse.FileType('r'))
    args = parser.parse_args()

    if args.names:
        names = parse_names(args.names)
    else:
        names = None

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    plot_cumrewards(args.infile, args.outdir)

    # Pandas approach.
    df = pd.read_csv(args.infile)

    plot_beliefs(df, args.outdir, formatter=str_state(names))
    plot_actions(df, args.outdir, formatter=str_action)
    plot_observations(df, args.outdir, formatter=str_observation)

    """
    # Record timing.
    end_time = time.clock()
    print 'took {:2f} seconds'.format(end_time - start_time)
    """
