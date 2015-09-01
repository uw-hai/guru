"""analyze.py

Analyze work_learn output.

TODO: Handle new model params format.
TODO: Parse model 'hyper' column, in place of alpha and beta.

"""

import multiprocessing as mp
import time
import os
import csv
import ast
import argparse
import collections
import itertools
import functools as ft
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
import util
import work_learn_problem as wlp

CI = 95  # Confidence interval

def tsplot_robust(df, time, unit, condition, value, ci):
    """Plot timeseries data with different x measurements and 95% ci."""
    if ci != 95:
        raise NotImplementedError
    n = df.groupby([condition, time]).count()[value].reset_index().pivot(index=time, columns=condition, values=value)
    # Use seaborn iff all conditions have the same number of measurements for
    # each time point.
    if len(n) == sum(n.duplicated()) + 1:
        return sns.tsplot(df, time=time, condition=condition,
                          unit=unit, value=value, ci=ci)
    else:
        means = df.groupby([condition, time], as_index=False)[value].mean().pivot(index=time, columns=condition, values=value)
        sem = df.groupby([condition, time], as_index=False)[value].aggregate(ss.sem).pivot(index=time, columns=condition, values=value)
        return means.plot(yerr=1.96 * sem)

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

def plot_beliefs(df, outfname, t_frac=1.0, formatter=None, line=False,
                 logx=True):
    df = df[df.t <= df.t.max() * t_frac].reset_index()

    df_b = pd.DataFrame(df.b.str.split().tolist()).astype(float)
    states = df_b.columns

    df = df.join(df_b)
    b_sums = df.groupby(['policy','t'], as_index=False).sum()
    if formatter:
        b_sums.rename(columns=dict((x, formatter(x)) for x in states), inplace=True)
        states = [formatter(x) for x in df_b.columns]
    for p, df_b in b_sums.groupby('policy', as_index=False):
        if not line:
            step_by_t(df_b).plot(x='t', y=states, kind='area',
                                 title='Belief counts', logx=logx)
        else:
            df_b.plot(x='t', y=states, kind='line',
                      title='Belief counts', logx=logx)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_b.to_csv(fname + '.csv', index=False)

def plot_actions(df, outfname, t_frac=1.0, formatter=None, line=False,
                 logx=True):
    df = df[df.t <= df.t.max() * t_frac]

    actions = df.groupby(['policy', 't'])['a'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        actions.rename(columns=dict((x, formatter(x)) for x in actions.columns[2:]), inplace=True)
    for p, df_a in actions.groupby('policy', as_index=False):
        if not line:
            step_by_t(df_a).plot(x='t', y=actions.columns[2:], kind='area',
                                 logx=logx)
        else:
            df_a.plot(x='t', y=actions.columns[2:], kind='line', logx=logx)
        plt.ylabel('Number of actions')
        plt.xlabel('Time')

        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_a.to_csv(fname + '.csv', index=False)

def plot_actions_subcount(df, outfname, actions_filter, ylabel):
    """Plot mean number of times the given actions are taken"""
    df = df[['iteration','policy','a']].copy()
    df = df[df['a'].map(lambda a: not np.isnan(a) and actions_filter(a))]
    df = df.groupby(['iteration','policy'])['a'].count().fillna(0.).reset_index()
    if len(df) == 0:
        return

    ax = sns.barplot('policy', y='a', data=df, ci=CI)

    # Save .csv file.
    s1 = df.groupby('policy')['a'].mean()
    s2 = df.groupby('policy')['a'].sem()
    df_means = pd.DataFrame({'count (mean)': s1,
                             'count (sem)': s2}).reset_index()
    df_means.to_csv(outfname + '.csv', index=False)

    # Record significance
    policies = df['policy'].unique()
    sigs = []
    for p1,p2 in itertools.combinations(policies, 2):
        v1 = df[df.policy == p1]['a']
        v2 = df[df.policy == p2]['a']
        t, pval = ss.ttest_ind(v1, v2)
        sigs.append({'policy1': p1,
                     'policy2': p2,
                     'tstat': t,
                     'pval': pval})
    df_sig = pd.DataFrame(sigs)
    df_sig.to_csv(outfname + '_sig.csv', index=False)

    # Finish plotting.
    ax.set_ylim(0, None)
    plt.ylabel(ylabel)
    plt.savefig(outfname + '.png')
    plt.close()

def plot_observations(df, outfname, t_frac=1.0, formatter=None, line=False,
                      logx=True):
    df = df[df.t <= df.t.max() * t_frac]

    obs = df.groupby(['policy', 't'])['o'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        obs.rename(columns=dict((x, formatter(x)) for x in obs.columns[2:]), inplace=True)
    for p, df_o in obs.groupby('policy', as_index=False):
        if not line:
            step_by_t(df_o).plot(x='t', y=obs.columns[2:], kind='area',
                                 title='Observation counts', logx=logx)
        else:
            df_o.plot(x='t', y=obs.columns[2:], kind='line',
                      title='Observation counts', logx=logx)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_o.to_csv(fname + '.csv', index=False)

def plot_reward_by_t(df, outfname):
    #df = df[df.t <= min(df.groupby(['policy','iteration'])['t'].max())]
    #print df.groupby(['policy','iteration'])['t'].max()
    #print df.groupby(['policy','iteration'])['t'].min()
    #print df.groupby(['policy', 'iteration']).count()
    #print '!!!!!'
    df.sort(['t'])
    df['reward'] = df['r'].fillna(0) + df['cost'].fillna(0)
    df['csr'] = df.groupby(['policy','iteration'])['reward'].cumsum()
    #ax = sns.tsplot(df, time='t', condition='policy', unit='iteration', value='csr', ci=CI)

    ax = df.groupby(['policy','t'], as_index=False)['csr'].mean().pivot(index='t', columns='policy', values='csr').plot()
    plt.ylabel('Cumulative reward')
    plt.xlabel('t')
    ax.set_xlim(0, None)
    plt.savefig(outfname + '.png')
    plt.close()
    #df.sort(['policy','iteration','episode']).to_csv(fname + '.csv',
    #                                                 index=False)

def plot_reward_by_budget(df, outfname):
    df.sort(['t'])
    df['r'] = df['r'].fillna(0)
    df['cum_r'] = df.groupby(['policy', 'iteration'])['r'].cumsum()
    df['cum_cost'] = -1 * df.groupby(['policy', 'iteration'])['cost'].cumsum()
    df = df.groupby(['policy', 'iteration', 'cum_cost'],
                    as_index=False)['cum_r'].last()
    #ax = df.groupby(['policy', 'cum_cost'], as_index=False)['cum_r'].mean().pivot(index='cum_cost', columns='policy', values='cum_r').plot()
    ax = tsplot_robust(df, time='cum_cost', condition='policy', unit='iteration', value='cum_r', ci=CI)

    plt.ylabel('Cumulative reward')
    plt.xlabel('Budget spent')
    ax.set_xlim(0, None)
    plt.savefig(outfname + '.png')
    plt.close()

def plot_timings(df_timings, outfname):
    for t in ('resolve', 'estimate'):
        df_filter = df_timings[df_timings['type'] == t]
        if len(df_filter.index) > 0:
            ax = tsplot_robust(df_filter, time='worker', unit='iteration',
                               condition='policy', value='duration', ci=CI)
            ax.set_ylim(0, None)
            ax.set_xlim(0, None)
            fname = outfname + '-{}'.format(t)
            plt.savefig(fname + '.png')
            plt.close()

# TODO: Move this to pomdp.py
def param_to_string(p):
    """Convert a param in tuple form, as in pomdp.py to a string.

    >>> param_to_string(('p_learn', None))
    'p_learn'
    >>> param_to_string(('p_guess', 2))
    'p_guess_w2'
    >>> param_to_string((('p_s', 2), None))
    'p_s2'
    >>> param_to_string((('p_s', 2), 1))
    'p_s2_w1'

    """
    if not isinstance(p, tuple):
        return p

    if isinstance(p[0], tuple):
        name = '{}{}'.format(*p[0])
    else:
        name = p[0]
    return name if p[1] is None else '{}_w{}'.format(name, p[1])

def plot_params(df_model, outfname):
    # Parse
    df_model['param'] = df_model['param'].apply(
        lambda x: param_to_string(ast.literal_eval(x)) if
                  x.startswith('(') else x)

    # Separate ground truth params
    df_gt = df_model[df_model.iteration.isnull()]
    df_est = df_model[df_model.iteration.notnull()]

    df_gt['v'] = df_gt['v'].apply(ast.literal_eval)
    df_est['v'] = df_est['v'].apply(ast.literal_eval)
    df_est['hyper'] = df_est['hyper'].apply(ast.literal_eval)

    # Find dist from true param.
    df_est = df_est.merge(df_gt, how='left', on='param', suffixes=('', '_t'))

    df_est['dist'] = np.subtract(
        df_est['v'].apply(lambda x: np.array(x)[:-1]),
        df_est['v_t'].apply(lambda x: np.array(x)[:-1]))
    df_est['dist_l1'] = df_est['dist'].apply(lambda x: np.linalg.norm(x, 1))
    df_est['dist_l2'] = df_est['dist'].apply(lambda x: np.linalg.norm(x, 2))

    for s in ['l1', 'l2']:
        df_means = df_est.groupby(['policy', 'iteration', 'worker'],
                                  as_index=False).mean()
        ax = tsplot_robust(df_means, time='worker', unit='iteration',
                           condition='policy', value='dist_{}'.format(s),
                           ci=CI)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, None)
        plt.ylabel('Mean distance ({}) from true parameter values'.format(s))
        plt.xlabel('worker')
        fname = outfname + '_dist_{}_mean'.format(s)
        plt.savefig(fname + '.png')
        plt.close()

    for p, df_p in df_est.groupby('policy', as_index=False):
        # BUG: Uses only first coordinate for each parameter.
        # TODO: Make this work for dirichlet by splitting the rows.
        df_p['v_1'] = df_p['v'].apply(lambda x: np.array(x)[0])
        ax = tsplot_robust(df_p, time='worker', unit='iteration',
                           condition='param', value='v_1', ci=CI)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, None)
        plt.ylabel('Estimated parameter value')
        plt.xlabel('Worker')
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()

        for s in ['l1', 'l2']:
            ax = tsplot_robust(df_p, time='worker', unit='iteration',
                               condition='param', value='dist_{}'.format(s),
                               ci=CI)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, None)
            plt.ylabel('Distance ({}) from true parameter value'.format(s))
            plt.xlabel('Worker')
            fname = outfname + '_dist_{}_p-{}'.format(s, p)
            plt.savefig(fname + '.png')
            plt.close()

        if np.all(df_p.hyper.notnull()):
            # BUG: Uses only first coordinate for each parameter.
            # TODO: Make this work for dirichlet by splitting the rows.
            df_p['dirichlet_mean'] = df_p['hyper'].apply(
                lambda x: ss.dirichlet.mean(x)[0])
            df_p['dirichlet_var_l1'] = df_p['hyper'].apply(
                lambda x: np.linalg.norm(ss.dirichlet.var(x), 1))
            df_p['dirichlet_mode'] = df_p['hyper'].apply(
                lambda x: util.dirichlet_mode(x)[0])

            for s in ['mean', 'var_l1', 'mode']:
                ax = tsplot_robust(
                    df_p, time='worker', unit='iteration',
                    condition='param', value='dirichlet_{}'.format(s), ci=CI)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, None)
                plt.ylabel('Parameter posterior {}'.format(s))
                plt.xlabel('Worker')
                fname = outfname + '_dirichlet_{}_p-{}'.format(s, p)
                plt.savefig(fname + '.png')
                plt.close()

def finished(df):
    """Filter for policy iterations that have used all budget."""
    #df.drop(df.tail(1).index, inplace=True)  # in case last row is incomplete
    #if len(df.index) > 0:
    #    df = df.join(df.groupby(['iteration','policy'])['episode'].max(), on=['iteration','policy'], rsuffix='_max')
    #    max_episode = df['episode'].max()
    #    df = df[~(df.episode_max < max_episode)]  # TODO: Double-check
    return df


def query_names_df(df, s, c, i):
    return df[(df.type == s) & (df.i == int(i))].reset_index()[c][0]

def make_plots(infiles, outdir, models=[], timings=[], names=None,
               policies=None, line=False, log=True, first_worker=False):
    """Make plots.

    Args:
        names:          Filepath to csv file mapping indices to strings
        log:            Use log scale for breakdown plots
        policies:       List of policies to include

    """
    action_uses_gold_question = None
    if names:
        df_names = pd.read_csv(names,
                               true_values=['True', 'true'],
                               false_values=['False', 'false'])
        str_action = ft.partial(query_names_df, df_names, 'action', 's')
        str_state = ft.partial(query_names_df, df_names, 'state', 's')
        str_observation = ft.partial(
            query_names_df, df_names, 'observation', 's')
        if 'uses_gold' in df_names.columns:
            action_uses_gold_question = ft.partial(query_names_df, df_names,
                                                   'action', 'uses_gold')
    else:
        str_action = lambda i: str(i)
        str_observation = lambda i: str(i)
        str_state = lambda i: str(i)

    util.ensure_dir(outdir)

    df = pd.concat([finished(pd.read_csv(f)) for f in infiles],
                   ignore_index=True)
    if policies is not None:
        df = df[df.policy.isin(policies)]
    if first_worker:
        df = df[df.worker == 0]

    if len(timings) > 0:
        df_timings = pd.concat([finished(pd.read_csv(f)) for f in timings],
                               ignore_index=True)
        if policies is not None:
            df_timings = df_timings[df_timings.policy.isin(policies)]
        if len(df_timings.index) > 0:
            df_timings = df_timings
            plot_timings(df_timings, os.path.join(outdir, 't'))
            print 'Done plotting timings'

    if action_uses_gold_question is not None:
        plot_actions_subcount(
            df, outfname=os.path.join(outdir, 'gold_questions_used'),
            actions_filter=action_uses_gold_question,
            ylabel='Mean number of gold questions used')

    for m in models:
        df_model = finished(pd.read_csv(m))
        plot_params(df_model, os.path.join(outdir, 'params'))
    print 'Done plotting params'

    plot_reward_by_t(df, os.path.join(outdir, 'r_t'))
    plot_reward_by_budget(df, os.path.join(outdir, 'r_cost'))
    plot_beliefs(df, os.path.join(outdir, 'b'),
                 formatter=str_state, line=line, logx=log)
    plot_actions(df, os.path.join(outdir, 'a'),
                 formatter=str_action, line=line, logx=log)
    plot_observations(df, os.path.join(outdir, 'o'),
                      formatter=str_observation, line=line, logx=log)

def main(filenames, policies=None, line=False, log=True,
         single=False, dest=None, first_worker=False):
    if single:
        if dest is None:
            raise Exception('Must specify a destination folder')

        names = os.path.splitext(filenames[0])[0] + '_names.csv'
        models = [os.path.splitext(f)[0] + '_model.csv' for f in filenames]
        timings = [os.path.splitext(f)[0] + '_timings.csv' for f in filenames]
        if not all(os.path.exists(t) for t in timings):
            timings = []

        make_plots(infiles=filenames,
                   outdir=dest,
                   names=names,
                   models=models,
                   timings=timings,
                   policies=policies,
                   line=line,
                   log=log,
                   first_worker=first_worker)
    else:
        jobs = []
        for f in filenames:
            # If result file is of the form 'f.end', assume directory also
            # contains 'f_model.csv' and 'f_names.csv'.
            # Output plots in a subdirectory with name 'f'.
            basename = os.path.basename(f)
            basename_no_ending = os.path.splitext(basename)[0]

            dirname = os.path.dirname(f)
            plotdir = os.path.join(dirname, basename_no_ending)
            util.ensure_dir(plotdir)

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
                policies=policies,
                line=line,
                log=log,
                first_worker=first_worker))
            jobs.append(p)
            p.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('result', type=str, nargs='+',
                        help='main experiment result .txt files')
    parser.add_argument('--line', dest='line', action='store_true',
                        help="Use line plots instead of area")
    parser.set_defaults(line=False)
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help="Don't use log for x-axis")
    parser.set_defaults(log=True)
    parser.add_argument('--single', dest='single', action='store_true',
                        help='Treat multiple inputs as single experiment')
    parser.set_defaults(single=False)
    parser.add_argument('--dest', '-d', type=str, help='Folder to store plots')
    parser.add_argument('--policies', type=str, nargs='*',
                        help='Policies to use')
    parser.add_argument('--first_worker', dest='first_worker',
                        action='store_true', help='Plot only first worker')
    parser.set_defaults(first_worker=False)
    args = parser.parse_args()

    main(filenames=args.result,
         policies=args.policies,
         line=args.line,
         log=args.log,
         single=args.single,
         dest=args.dest,
         first_worker=args.first_worker)
