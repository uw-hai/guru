"""analyze.py

Analyze work_learn output.

"""

import multiprocessing as mp
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

reserved_actions = ['ASK', 'EXP', 'NOEXP', 'BOOT']

def str_action(a):
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

def plot_beliefs(df, outfname, t_frac=1.0, formatter=None, logx=True):
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
                             title='Belief counts', logx=logx)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_b.to_csv(fname + '.csv')

def plot_actions(df, outfname, t_frac=1.0, formatter=None, logx=True):
    df = df[df.t <= df.t.max() * t_frac]

    actions = df.groupby(['policy', 't'])['a'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        actions.rename(columns=dict((x, formatter(x)) for x in actions.columns[2:]), inplace=True)
    for p, df_a in actions.groupby('policy', as_index=False):
        step_by_t(df_a).plot(x='t', y=actions.columns[2:], kind='area',
                             title='Action counts', logx=logx)

        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_a.to_csv(fname + '.csv')

def plot_observations(df, outfname, t_frac=1.0, formatter=None, logx=True):
    df = df[df.t <= df.t.max() * t_frac]

    obs = df.groupby(['policy', 't'])['o'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        obs.rename(columns=dict((x, formatter(x)) for x in obs.columns[2:]), inplace=True)
    for p, df_o in obs.groupby('policy', as_index=False):
        step_by_t(df_o).plot(x='t', y=obs.columns[2:], kind='area',
                             title='Observation counts', logx=logx)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_o.to_csv(fname + '.csv')

def plot_reward_by_episode(df, outfname):
    df = df[['iteration','policy','episode','r']]
    df = df.groupby(['iteration','policy','episode']).sum().fillna(0).reset_index()[['iteration','policy','episode','r']]

    df['csr'] = df.sort(['episode']).groupby(['iteration','policy'])['r'].cumsum()
    ax = sns.tsplot(df, time='episode', condition='policy', unit='iteration', value='csr', ci=CI, interpolate=df['episode'].max() >= INTERPOLATE_MIN)
    plt.savefig(outfname + '.png')
    plt.close()
    #df.sort(['iteration','policy']).to_csv(fname + '.csv')

    # Plot bar version. Only useful for non-RL.
    # Select only certain policies for printing.
    #df = df[(df.policy == 'pbvi-h10') | (df.policy == 'pbvi-h50')]
    df['it-ep'] = df['iteration'].map(str) + ',' + df['episode'].map(str)
    df_sums = df.groupby(['policy', 'it-ep'], as_index=False)['r'].sum()
    ax = sns.barplot('policy', y='r', data=df_sums, ci=CI)
    plt.savefig(outfname + '_bar.png')
    plt.close()

def plot_reward_by_t(df, outfname):
    df = df[['policy','iteration','episode','t','r']]

    # Fill out table.
    df = pd.pivot_table(df, index=['iteration','episode'], columns=['policy','t'], values=['r']).stack(level=['policy','t'], dropna=False).fillna(0).reset_index()

    df['csr'] = df.sort(['t']).groupby(['policy','iteration','episode'])['r'].cumsum()
    df['it-ep'] = df['iteration'].map(str) + ',' + df['episode'].map(str)
    ax = sns.tsplot(df, time='t', condition='policy', unit='it-ep', value='csr', ci=CI, interpolate=df['t'].max() >= INTERPOLATE_MIN)
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

def plot_params(df_model, outfname):
    # Separate ground truth params
    df_gt = df_model[df_model.iteration.isnull()]
    df_est = df_model[df_model.iteration.notnull()]

    # Find dist from true param.
    df_est = df_model.merge(df_gt, how='left', on='param', suffixes=('', '_t'))
    df_est['dist'] = np.abs(df_est['v'] - df_est['v_t'])

    for p, df_p in df_est.groupby('policy', as_index=False):
        sns.tsplot(df_p, time='episode', unit='iteration', condition='param', value='v', ci=CI)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()

        sns.tsplot(df_p, time='episode', unit='iteration', condition='param', value='dist', ci=CI)
        fname = outfname + '_dist_p-{}'.format(p)
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
    df = df[~(df.episode_max < max_episode)]  # TODO: Double-check
    return df, max_episode

def make_plots(infile, outdir, model=None, names=None,
               episode_step=100, exclude=[], noexp=True, log=True):
    """Make plots.

    Args:
        episode_step:   Episode step size for plotting t on x-axis
        noexp:          Plot NOEXP actions
        log:            Use log scale for breakdown plots
        exclude:        List of policies to exclude

    """
    if names:
        with open(names, 'r') as f:
            names = parse_names(f)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    df, max_episode = finished(pd.read_csv(infile))
    if not noexp:
        df = df[df.a != reserved_actions.index('NOEXP')]
    if exclude:
        for p in exclude:
            df = df[df.policy != p]

    plot_reward_by_episode(df, os.path.join(outdir, 'r'))
    plot_solve_t_by_episode(df, os.path.join(outdir, 't'))

    episode_pairs = range(0, max_episode + 2, episode_step)
    for p in zip(episode_pairs[:-1], episode_pairs[1:]):
        df_filter = df[(df.episode >= p[0]) & (df.episode < p[1])]
        e_str = 'e{}-{}'.format(*p)
        plot_reward_by_t(df_filter, os.path.join(outdir, 'r_t_' + e_str))
        plot_beliefs(df_filter, os.path.join(outdir, 'b_' + e_str), formatter=str_state(names), logx=log)
        plot_actions(df_filter, os.path.join(outdir, 'a_' + e_str), formatter=str_action, logx=log)
        plot_observations(df_filter, os.path.join(outdir, 'o_' + e_str), formatter=str_observation, logx=log)

    if model is not None:
        df_model = pd.read_csv(model)
        if len(df_model) > 0:
            df_model, _ = finished(df_model)
            plot_params(df_model, os.path.join(outdir, 'params'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('config', type=str, nargs='+', help='config files')
    parser.add_argument('--episode_step', type=int, default=100)
    parser.add_argument('--exclude', type=str, nargs='*', help='Policies to exclude')
    args = parser.parse_args()
    
    jobs = []
    for f in args.config:
        basename = os.path.basename(f)
        name_exp = os.path.splitext(basename)[0]

        infile = os.path.join('res', '{}.txt'.format(name_exp))
        names = os.path.join('res', '{}_names.csv'.format(name_exp))
        model = os.path.join('res', '{}_model.csv'.format(name_exp))
        outdir = os.path.join('res', name_exp)

        p = mp.Process(target=make_plots, kwargs=dict(
            infile=infile,
            outdir=outdir,
            names=names,
            model=model,
            episode_step=args.episode_step,
            exclude=args.exclude))
        jobs.append(p)
        p.start()

