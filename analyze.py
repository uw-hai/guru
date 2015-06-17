"""analyze.py

Analyze work_learn output.

"""

import multiprocessing as mp
import time
import os
import csv
import argparse
import collections
import itertools
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
from util import ensure_dir

INTERPOLATE_MIN = 11
CI = 95  # Confidence interval

#reserved_actions = ['ASK', 'EXP', 'NOEXP', 'BOOT']
reserved_actions = ['Work', 'Teach', "Don't teach", 'Boot']
def action_uses_gold_question(a):
    return a >= len(reserved_actions)

def str_action(a):
    N_RES_A = len(reserved_actions)
    a = int(a)
    if a < N_RES_A:
        s = reserved_actions[a]
    else:
        #s = "Q{}".format(a - N_RES_A)
        s = "Test {}".format(a - N_RES_A + 1)
    #return '{} ({})'.format(a, s)
    return s

def str_state(d=None):
    if d == None:
        return lambda s: s
    return lambda s: '{} ({})'.format(s, d['state', s])


def str_observation(o):
    o_names = ['WRONG', 'RIGHT', 'TERM']
    o = int(o)
    return '{} ({})'.format(o, o_names[o])

def step_by_t(df, interval=0.1):
    """Copy data from t=i to the range t=[i, i+1) at the given interval.
    
    Area plots look jagged, so this makes the lines more vertical.

    """
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
        df_b.to_csv(fname + '.csv', index=False)

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
        df_a.to_csv(fname + '.csv', index=False)

def plot_actions_subcount(df, outfname, actions_filter, ylabel):
    """Plot mean number of times the given actions are taken"""
    df = df[['iteration','policy','episode','a']].copy()
    df['a'] = df['a'].map(lambda a: a if actions_filter(a) else float('NaN'))
    df = df.groupby(['iteration','policy','episode'])['a'].count().fillna(0.).reset_index()

    if df['episode'].max() > 0:
        ax = sns.tsplot(
            df, time='episode', condition='policy', unit='iteration', value='a',
            ci=CI, interpolate=df['episode'].max() >= INTERPOLATE_MIN)
        plt.xlabel('Episode')
        ax.set_xlim(0, None)
    else:
        ax = sns.barplot('policy', y='a', data=df, ci=CI)

        # Save .csv file.
        s1 = df.groupby('policy')['a'].mean()
        s2 = df.groupby('policy')['a'].sem()
        df_means = pd.DataFrame({'count (mean)': s1,
                                 'count (sem)': s2}).reset_index()
        df_means.to_csv(outfname + '.csv', index=False)

    # Finish plotting.
    ax.set_ylim(0, None)
    plt.ylabel(ylabel)
    plt.savefig(outfname + '.png')
    plt.close()

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
        df_o.to_csv(fname + '.csv', index=False)

def plot_reward(df, outfname, estimator=np.mean):
    df = df[['iteration','policy','episode','r']]
    df = df.groupby(['iteration','policy','episode']).sum().fillna(0).reset_index()[['iteration','policy','episode','r']]

    ax = sns.barplot('policy', y='r', data=df, estimator=estimator, ci=CI)
    plt.ylabel('Reward ({})'.format(estimator.__name__))
    fname = outfname + '_bar'
    estimator_name = estimator.__name__
    if estimator_name != 'mean':
        fname += '_{}'.format(estimator_name)
    plt.savefig(fname + '.png')
    plt.close()

    if estimator_name == 'mean':
        s1 = df.groupby('policy')['r'].mean()
        s2 = df.groupby('policy')['r'].sem()
        df_means = pd.DataFrame({'reward (mean)': s1,
                                 'reward (sem)': s2}).reset_index()
        df_means.to_csv(fname + '.csv', index=False)

        # Record significance
        policies = df['policy'].unique()
        sigs = []
        for p1,p2 in itertools.combinations(policies, 2):
            v1 = df[df.policy == p1]['r']
            v2 = df[df.policy == p2]['r']
            t, pval = ss.ttest_ind(v1, v2)
            sigs.append({'policy1': p1,
                         'policy2': p2,
                         'tstat': t,
                         'pval': pval})
        df_sig = pd.DataFrame(sigs)
        df_sig.to_csv(fname + '_sig.csv', index=False)



def plot_reward_by_episode(df, outfname):
    df = df[['iteration','policy','episode','r']]
    df = df.groupby(['iteration','policy','episode']).sum().fillna(0).reset_index()[['iteration','policy','episode','r']]

    ax = sns.tsplot(df, time='episode', condition='policy', unit='iteration', value='r',
               ci=CI, interpolate=df['episode'].max() >= INTERPOLATE_MIN)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    ax.set_xlim(0, None)
    plt.savefig(outfname + '.png')
    plt.close()

    df['csr'] = df.sort(['episode']).groupby(
        ['iteration','policy'])['r'].cumsum()
    ax = sns.tsplot(
        df, time='episode', condition='policy', unit='iteration', value='csr',
        ci=CI, interpolate=df['episode'].max() >= INTERPOLATE_MIN)
    ax.set_xlim(0, None)
    plt.ylabel('Cumulative reward')
    plt.xlabel('Episode')
    plt.savefig(outfname + '_cum.png')
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
    #df.sort(['policy','iteration','episode']).to_csv(fname + '.csv',
    #                                                 index=False)


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

def plot_timings(df_timings, outfname):
    if df_timings['episode'].max() > 1:
        for t in ('resolve', 'estimate'):
            df_filter = df_timings[df_timings['type'] == t]
            if len(df_filter.index) > 0:
                ax = sns.tsplot(df_filter, time='episode', unit='iteration', condition='policy', value='duration', ci=CI)
                ax.set_ylim(0, None)
                ax.set_xlim(0, None)
                fname = outfname + '-{}'.format(t)
                plt.savefig(fname + '.png')
                plt.close()
    else:
        df_filter = df_timings[df_timings['type'] == 'resolve']
        if len(df_filter.index) > 0:
            sns.barplot('policy', y='duration', data=df_filter, ci=CI)
            fname = outfname + '-resolve'
            plt.savefig(fname + '.png')
            plt.close()

def plot_params(df_model, outfname):
    # Separate ground truth params
    df_gt = df_model[df_model.iteration.isnull()]
    df_est = df_model[df_model.iteration.notnull()]

    # Find dist from true param.
    df_est = df_est.merge(df_gt, how='left', on='param', suffixes=('', '_t'))
    df_est['dist'] = np.abs(df_est['v'] - df_est['v_t'])

    if np.all(df_est.alpha.notnull()):
        df_est['beta_mean'] = df_est.apply(
            lambda r: ss.beta.mean(r['alpha'], r['beta']), axis=1)
        df_est['beta_std'] = df_est.apply(
            lambda r: ss.beta.std(r['alpha'], r['beta']), axis=1)
        df_est['beta_mode'] = df_est.apply(
            lambda r: (r['alpha'] - 1) / (r['alpha'] + r['beta'] - 2), axis=1)

    df_est['it-param'] = df_est['iteration'].map(str) + ',' + \
                         df_est['param'].map(str)

    for p, df_p in df_est.groupby('policy', as_index=False):
        ax = sns.tsplot(df_p, time='episode', unit='iteration',
                        condition='param', value='v', ci=CI)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, None)
        plt.ylabel('Estimated parameter value')
        plt.xlabel('Episode')
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()

        ax = sns.tsplot(df_p, time='episode', unit='iteration',
                        condition='param', value='dist', ci=CI)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, None)
        plt.ylabel('Distance from true parameter value')
        plt.xlabel('Episode')
        fname = outfname + '_dist_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()

        ax = sns.tsplot(df_p, time='episode', unit='it-param',
                        value='dist', ci=CI)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, None)
        plt.ylabel('Mean distance from true parameter values')
        plt.xlabel('Episode')
        fname = outfname + '_dist_mean_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()

        if np.all(df_p.alpha.notnull()):
            for s in ['mean', 'std', 'mode']:
                ax = sns.tsplot(df_p, time='episode', unit='iteration',
                                condition='param', value='beta_{}'.format(s),
                                ci=CI)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, None)
                plt.ylabel('Parameter posterior {}'.format(s))
                plt.xlabel('Episode')
                fname = outfname + '_beta_{}_p-{}'.format(s, p)
                plt.savefig(fname + '.png')
                plt.close()

def parse_names(f):
    """Return dictionary mapping type and index to a string"""
    d = dict()
    dr = csv.DictReader(f)
    for r in dr:
        d[r['type'],int(r['i'])] = r['s']
    return d

def finished(df):
    """Filter for policy iterations that have completed all episodes."""
    #df.drop(df.tail(1).index, inplace=True)  # in case last row is incomplete
    if len(df.index) > 0:
        df = df.join(df.groupby(['iteration','policy'])['episode'].max(), on=['iteration','policy'], rsuffix='_max')
        max_episode = df['episode'].max()
        df = df[~(df.episode_max < max_episode)]  # TODO: Double-check
    return df

def expand_episodes(df):
    """Make all policies have the same max episode by copying data"""
    max_episodes = dict()
    for p, df_p in df.groupby('policy'):
        max_episodes[p] = df_p['episode'].max()
    max_episode = max(max_episodes.itervalues())
    if not all(v == 0 or v == max_episode for v in max_episodes.itervalues()):
        raise Exception('Cannot expand different max episodes')
    elif not max_episode >= 0:
        raise Exception('Invalid max episode')
    elif max_episode == 0:
        return df

    # Expand.
    dfs = []
    for p, df_p in df.groupby('policy'):
        if max_episodes[p] > 0:
            dfs.append(df_p)
        else:
            dfs_expanded = []
            for i in xrange(max_episode + 1):
                df_new = df_p.copy(deep=True)
                df_new['episode'] = i
                dfs_expanded.append(df_new)
            dfs += dfs_expanded
    return pd.concat(dfs, ignore_index=True)

def make_plots(infiles, outdir, models=[], timings=[], names=None,
               episode_step=100, policies=None, fast=False,
               noexp=True, log=True):
    """Make plots.

    Args:
        episode_step:   Episode step size for plotting t on x-axis
        noexp:          Plot NOEXP actions
        log:            Use log scale for breakdown plots
        policies:       List of policies to include
        fast:           Don't make plots with time as x-axis.

    """
    if names:
        with open(names, 'r') as f:
            names = parse_names(f)

    ensure_dir(outdir)

    df = pd.concat([finished(pd.read_csv(f)) for f in infiles],
                   ignore_index=True)
    if policies is not None:
        df = df[df.policy.isin(policies)]
    max_episode = df.episode.max()
    df = expand_episodes(df)
    if not noexp:
        df = df[df.a != reserved_actions.index("Don't teach")]

    if len(timings) > 0:
        df_timings = pd.concat([finished(pd.read_csv(f)) for f in timings],
                               ignore_index=True)
        if policies is not None:
            df_timings = df_timings[df_timings.policy.isin(policies)]
        df_timings = expand_episodes(df_timings)
        plot_timings(df_timings, os.path.join(outdir, 't'))
    print 'Done plotting timings'

    plot_actions_subcount(
        df, outfname=os.path.join(outdir, 'gold_questions_used'),
        actions_filter=action_uses_gold_question,
        ylabel='Mean number of gold questions used')

    if max_episode > 0:
        # Make time series plots with episode as x-axis.
        plot_reward_by_episode(df, os.path.join(outdir, 'r'))
        plot_solve_t_by_episode(df, os.path.join(outdir, 't'))
        print 'Done plotting reward and solve_t by episode'
        for m in models:
            df_model = finished(pd.read_csv(m))
            if df_model['episode'].max() > 0:
                plot_params(df_model, os.path.join(outdir, 'params'))
        print 'Done plotting params'
    else:
        plot_reward(df, os.path.join(outdir, 'r'))
        plot_reward(df, os.path.join(outdir, 'r'), estimator=np.std)
        plot_reward(df, os.path.join(outdir, 'r'), estimator=ss.variation)
        print 'Done plotting reward'

    #episode_pairs = range(0, max_episode + 2, min(max_episode + 1, episode_step))
    episodes_in_detail = range(0, max_episode + 1, episode_step)
    if episodes_in_detail[-1] != max_episode:
        episodes_in_detail.append(max_episode)
    if not fast:
        #for p in zip(episode_pairs[:-1], episode_pairs[1:]):
        for e in episodes_in_detail:
            #df_filter = df[(df.episode >= p[0]) & (df.episode < p[1])]
            df_filter = df[df.episode == e]
            #e_str = 'e{}-{}'.format(*p)
            e_str = 'e{}'.format(e)
            plot_reward_by_t(df_filter, os.path.join(outdir, 'r_t_' + e_str))
            plot_beliefs(df_filter, os.path.join(outdir, 'b_' + e_str),
                         formatter=str_state(names), logx=log)
            plot_actions(df_filter, os.path.join(outdir, 'a_' + e_str),
                         formatter=str_action, logx=log)
            plot_observations(df_filter, os.path.join(outdir, 'o_' + e_str),
                              formatter=str_observation, logx=log)
            print 'Done plotting episode {} in detail'.format(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('result', type=str, nargs='+',
                        help='main experiment result .txt files')
    parser.add_argument('--episode_step', type=int, default=10)
    parser.add_argument('--no-noexp', dest='noexp', action='store_false',
                        help="Don't print NOEXP actions.")
    parser.set_defaults(noexp=True)
    parser.add_argument('--single', dest='single', action='store_true',
                        help='Treat multiple inputs as single experiment')
    parser.set_defaults(single=False)
    parser.add_argument('--fast', dest='fast', action='store_true',
                        help="Don't make plots with time as x-axis")
    parser.set_defaults(fast=False)
    parser.add_argument('--dest', '-d', type=str, help='Folder to store plots')
    parser.add_argument('--policies', type=str, nargs='*',
                        help='Policies to use')
    args = parser.parse_args()

    if args.single:
        if args.dest is None:
            raise Exception('Must specify a destination folder')
        if args.dest:
            plotdir = args.dest

        names = os.path.splitext(args.result[0])[0] + '_names.csv'
        models = [os.path.splitext(f)[0] + '_model.csv' for f in args.result]
        timings = [os.path.splitext(f)[0] + '_timings.csv' for f in args.result]
        if not all(os.path.exists(t) for t in timings):
            timings = []

        make_plots(infiles=args.result,
                   outdir=plotdir,
                   names=names,
                   models=models,
                   timings=timings,
                   episode_step=args.episode_step,
                   policies=args.policies,
                   fast=args.fast,
                   noexp=args.noexp)
    else:
        jobs = []
        for f in args.result:
            # If result file is of the form 'f.end', assume directory also
            # contains 'f_model.csv' and 'f_names.csv'.
            # Output plots in a subdirectory with name 'f'.
            basename = os.path.basename(f)
            basename_no_ending = os.path.splitext(basename)[0]

            dirname = os.path.dirname(f)
            plotdir = os.path.join(dirname, basename_no_ending)
            ensure_dir(plotdir)

            names = os.path.join(
                dirname, '{}_names.csv'.format(basename_no_ending))
            model = os.path.join(
                dirname, '{}_model.csv'.format(basename_no_ending))
            timings = os.path.join(
                dirname, '{}_timings.csv'.format(basename_no_ending))
            if os.path.exists(timings):
                timings = [timings]
            else:
                timings = []

            
            p = mp.Process(target=make_plots, kwargs=dict(
                infiles=[f],
                outdir=plotdir,
                names=names,
                models=[model],
                timings=timings,
                episode_step=args.episode_step,
                policies=args.policies,
                fast=args.fast,
                noexp=args.noexp))
            jobs.append(p)
            p.start()
