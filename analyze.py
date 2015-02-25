"""analyze.py

Analyze work_learn output.

"""

import time
import os
import csv
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns

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

def plot_beliefs(df, outdir, t_frac=1.0):
    df = df[df.t <= df.t.max() * t_frac]

    df_b = pd.DataFrame(df.b.str.split().tolist()).astype(float)
    states = df_b.columns

    df = df.join(df_b)
    b_sums = df.groupby(['policy','t'], as_index=False).sum()
    for p, df_b in b_sums.groupby('policy', as_index=False):
        df_b.plot(x='t', y=states, kind='area',
                  title='Belief counts', logx=True)
        fname = os.path.join(outdir, 'b_{}'.format(p))
        plt.savefig(fname + '.png')
        plt.close()
        df_b.to_csv(fname + '.csv')

def plot_actions(df, outdir, t_frac=1.0):
    df = df[df.t <= df.t.max() * t_frac]

    actions = df.groupby(['policy', 't'])['a'].value_counts().unstack().fillna(0.).reset_index()
    for p, df_a in actions.groupby('policy', as_index=False):
        df_a.plot(x='t', y=actions.columns[2:], kind='area',
                  title='Action counts', logx=True)
        fname = os.path.join(outdir, 'a_{}'.format(p))
        plt.savefig(fname + '.png')
        plt.close()
        df_a.to_csv(fname + '.csv')

def plot_observations(df, outdir, t_frac=1.0):
    df = df[df.t <= df.t.max() * t_frac]

    obs = df.groupby(['policy', 't'])['o'].value_counts().unstack().fillna(0.).reset_index()
    for p, df_o in obs.groupby('policy', as_index=False):
        df_o.plot(x='t', y=obs.columns[2:], kind='area',
                  title='Observation counts', logx=True)
        fname = os.path.join(outdir, 'o_{}'.format(p))
        plt.savefig(fname + '.png')
        plt.close()
        df_o.to_csv(fname + '.csv')


if __name__ == '__main__':
    start_time = time.clock()

    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'))
    parser.add_argument('-o', '--outdir', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    plot_cumrewards(args.infile, args.outdir)

    # Pandas approach.
    df = pd.read_csv(args.infile)

    plot_beliefs(df, args.outdir)
    plot_actions(df, args.outdir)
    plot_observations(df, args.outdir)

    """
    # Record timing.
    end_time = time.clock()
    print 'took {:2f} seconds'.format(end_time - start_time)
    """
