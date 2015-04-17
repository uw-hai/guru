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

INTERPOLATE_MIN = 11
CI = 95  # Confidence interval

def str_action(a):
    reserved_actions = ['ASK', 'EXP', 'NOEXP', 'BOOT']
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

def plot_beliefs(df, outfname, t_frac=1.0, formatter=None):
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
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_b.to_csv(fname + '.csv')

def plot_actions(df, outfname, t_frac=1.0, formatter=None):
    df = df[df.t <= df.t.max() * t_frac]

    actions = df.groupby(['policy', 't'])['a'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        actions.rename(columns=dict((x, formatter(x)) for x in actions.columns[2:]), inplace=True)
    for p, df_a in actions.groupby('policy', as_index=False):
        step_by_t(df_a).plot(x='t', y=actions.columns[2:], kind='area',
                             title='Action counts', logx=True)

        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_a.to_csv(fname + '.csv')

def plot_observations(df, outfname, t_frac=1.0, formatter=None):
    df = df[df.t <= df.t.max() * t_frac]

    obs = df.groupby(['policy', 't'])['o'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        obs.rename(columns=dict((x, formatter(x)) for x in obs.columns[2:]), inplace=True)
    for p, df_o in obs.groupby('policy', as_index=False):
        step_by_t(df_o).plot(x='t', y=obs.columns[2:], kind='area',
                             title='Observation counts', logx=True)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_o.to_csv(fname + '.csv')

def plot_reward_by_episode(df, outfname):
    df = df[['iteration','policy','episode','r']]
    df = df.groupby(['iteration','policy','episode']).sum().fillna(0).reset_index()[['iteration','policy','episode','r']]

    df['csr'] = df.sort(['episode']).groupby(['iteration','policy'])['r'].cumsum()
    sns.tsplot(df, time='episode', condition='policy', unit='iteration', value='csr', ci=CI, interpolate=df['episode'].max() >= INTERPOLATE_MIN)
    plt.savefig(outfname + '.png')
    plt.close()
    #df.sort(['iteration','policy']).to_csv(fname + '.csv')

def plot_reward_by_t(df, outfname):
    df = df[['policy','iteration','episode','t','r']]

    # Fill out table.
    df = pd.pivot_table(df, index=['iteration','episode'], columns=['policy','t'], values=['r']).stack(level=['policy','t'], dropna=False).fillna(0).reset_index()

    df['csr'] = df.sort(['t']).groupby(['policy','iteration','episode'])['r'].cumsum()
    df['it-ep'] = df['iteration'].map(str) + ',' + df['episode'].map(str)
    sns.tsplot(df, time='t', condition='policy', unit='it-ep', value='csr', ci=CI, interpolate=df['t'].max() >= INTERPOLATE_MIN)
    plt.savefig(outfname + '.png')
    plt.close()
    #df.sort(['policy','iteration','episode']).to_csv(fname + '.csv')

def plot_solve_t_by_episode(df, outfname):
    df = df[['iteration','policy','episode','sys_t']]
    df = df.sort('episode')
    df = df.groupby(['iteration','policy','episode']).first().reset_index()[['iteration','policy','episode','sys_t']]

    # Subtract start time for first episode.
    df = df.join(df.groupby(['iteration','policy'])['sys_t'].first(), on=['iteration','policy'], rsuffix='_start')
    df['elapsed time'] = df['sys_t'] - df['sys_t_start']

    sns.tsplot(df, time='episode', condition='policy', unit='iteration', value='elapsed time', ci=CI, interpolate=df['episode'].max() >= INTERPOLATE_MIN)
    plt.savefig(outfname + '.png')
    plt.close()

def model_cleanup(df):
    #df['a1'] = df['a1'].map(formatter)

    # Set shared group order for initial to state id.
    df.ix[df.param_type == 'initial', 'shared_o'] = df['a1']

    # Fill a2 and a3 for initial.
    df = df.fillna(-1)
    df['a1'] = df['a1'].astype(int)  # had NaN?
    df['a2'] = df['a2'].astype(int)  # had NaN
    df['a3'] = df['a3'].astype(int)  # had NaN
    df['type'] = df['param_type'] + '(' + df['a1'].map(str) + ',' + df['a2'].map(str) + ',' + df['a3'].map(str) + ')'
    return df

def plot_params(df_model_t, df_model, outfname, formatter):
    # Ignore rewards and clamped probabilities in true model.
    df_model_t = df_model_t[df_model_t.param_type != 'reward']
    if 'clamped' in df_model_t:
        df_model_t = df_model_t[~df_model_t.clamped.astype(bool)]

    df_t = model_cleanup(df_model_t)
    # One parameter from each shared group.
    df_t = df_t.groupby(['param_type','shared_g','shared_o']).first().reset_index()
    df = model_cleanup(df_model)
    df_merged = pd.merge(df, df_t, on=['type'], how='inner', suffixes=('_l', '_r'))
    # Distance from true param.
    df_merged['dist'] = np.abs(df_merged['v_l'] - df_merged['v_r'])

    for p, df_p in df_merged.groupby('policy', as_index=False):
        # To plot actual params, use value='v_l' instead of value='dist'.
        sns.tsplot(df_p, time='episode', unit='iteration', condition='type', value='dist', ci=CI)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()

    # Plot aggregate learned model all policies.
    # TODO: Refactor to avoid code duplication.
    df_merged['it-pol'] = df_merged['iteration'].map(str) + ',' + df_merged['policy'].map(str)
    sns.tsplot(df_merged, time='episode', unit='it-pol', condition='type', value='dist', ci=CI)
    plt.savefig(fname + '.png')
    plt.close()

def parse_names(f):
    d = dict()
    dr = csv.DictReader(f)
    for r in dr:
        d[r['type'],int(r['i'])] = r['s']
    return d

def finished(df):
    """Filter for policy iterations that have completed all episodes."""
    df.drop(df.tail(1).index, inplace=True)  # in case last row is incomplete
    df = df.join(df.groupby(['iteration','policy'])['episode'].max(), on=['iteration','policy'], rsuffix='_max')
    max_episode = df['episode'].max()
    df = df[df.episode_max == max_episode]
    return df, max_episode

if __name__ == '__main__':
    start_time = time.clock()

    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'))
    parser.add_argument('-o', '--outdir', type=str)
    parser.add_argument('-n', '--names', type=argparse.FileType('r'))
    parser.add_argument('--model_true', type=argparse.FileType('r'))
    parser.add_argument('--model_est', type=argparse.FileType('r'))
    parser.add_argument('--episode_step', type=int, default=100, help='Episode step size for plotting t on x-axis')
    args = parser.parse_args()

    if args.names:
        names = parse_names(args.names)
    else:
        names = None

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    df, max_episode = finished(pd.read_csv(args.infile))

    episode_pairs = range(0, max_episode, args.episode_step) + [max_episode + 1]
    for p in zip(episode_pairs[:-1], episode_pairs[1:]):
        df_filter = df[(df.episode >= p[0]) & (df.episode < p[1])]
        e_str = 'e{}-{}'.format(*p)
        plot_reward_by_t(df_filter, os.path.join(args.outdir, 'r_t_' + e_str))
        plot_beliefs(df_filter, os.path.join(args.outdir, 'b_' + e_str), formatter=str_state(names))
        plot_actions(df_filter, os.path.join(args.outdir, 'a_' + e_str), formatter=str_action)
        plot_observations(df_filter, os.path.join(args.outdir, 'o_' + e_str), formatter=str_observation)

    plot_reward_by_episode(df, os.path.join(args.outdir, 'r'))
    plot_solve_t_by_episode(df, os.path.join(args.outdir, 't'))

    if args.model_true and args.model_est:
        df_model_t = pd.read_csv(args.model_true)
        df_model = pd.read_csv(args.model_est)
        if len(df_model) > 0:
            df_model, _ = finished(df_model)
            plot_params(df_model_t, df_model, os.path.join(args.outdir, 'params'), formatter=str_state(names))

    """
    # Record timing.
    end_time = time.clock()
    print 'took {:2f} seconds'.format(end_time - start_time)
    """
