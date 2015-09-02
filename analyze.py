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
    """Plot timeseries data with different x measurements"""
    n = df.groupby([condition, time]).count()[value].reset_index().pivot(index=time, columns=condition, values=value)
    means = df.groupby([condition, time], as_index=False)[value].mean().pivot(index=time, columns=condition, values=value)
    sem = df.groupby([condition, time], as_index=False)[value].aggregate(ss.sem).pivot(index=time, columns=condition, values=value)

    # Use seaborn iff all conditions have the same number of measurements for
    # each time point.
    if len(n) == 0:
        raise Exception('Unable to plot empty dataframe')
    if len(n) == 1:
        return sns.barplot(condition, y=value, data=df, ci=CI)
    elif len(n) == sum(n.duplicated()) + 1:
        return sns.tsplot(df, time=time, condition=condition,
                          unit=unit, value=value, ci=ci)
    else:
        if ci != 95:
            raise NotImplementedError
        ax = plt.gca()
        for col in means.columns:
            line, = plt.plot(means.index, means[col], label=col)
            ax.fill_between(means.index,
                            means[col] - 1.96 * sem[col],
                            means[col] + 1.96 * sem[col],
                            facecolor=line.get_color(),
                            alpha=0.5,
                            where=np.isfinite(sem[col]))
        plt.legend()
        return ax

def step_by_col(df, col, interval=0.1):
    """Copy data from t=i to the range t=[i, i+1) at the given interval.
    
    Area plots look jagged, so this makes the lines more vertical.

    """
    dfs = []
    for i in np.arange(0, 1, interval):
        df2 = df.copy()
        df2[col] = df2[col].map(lambda x: x + i)
        dfs.append(df2)
    df_out = pd.concat(dfs, ignore_index=True)
    df_out.sort(['policy', col], inplace=True)
    df_out.reset_index()
    return df_out

def plot_beliefs(df, outfname, formatter=None, line=False, logx=True):
    df_b = pd.DataFrame(df.b.str.split().tolist()).astype(float)
    states = df_b.columns

    df = df.join(df_b)
    b_sums = df.groupby(['policy','t_rel'], as_index=False).sum()
    if formatter:
        b_sums.rename(columns=dict((x, formatter(x)) for x in states),
                      inplace=True)
        states = [formatter(x) for x in df_b.columns]
    for p, df_b in b_sums.groupby('policy', as_index=False):
        if not line:
            step_by_col(df_b, 't_rel').plot(x='t_rel', y=states, kind='area',
                                 title='Belief counts', logx=logx)
        else:
            df_b.plot(x='t_rel', y=states, kind='line',
                      title='Belief counts', logx=logx)
        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_b.to_csv(fname + '.csv', index=False)

def plot_actions(df, outfname, formatter=None, line=False, logx=True):
    actions = df.groupby(['policy', 't_rel'])['a'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        actions.rename(columns=dict((x, formatter(x)) for x in actions.columns[2:]), inplace=True)
    for p, df_a in actions.groupby('policy', as_index=False):
        if not line:
            step_by_col(df_a, 't_rel').plot(x='t_rel', y=sorted(actions.columns[2:]),
                                 kind='area', logx=logx)
        else:
            df_a.plot(x='t_rel', y=sorted(actions.columns[2:]),
                      kind='line', logx=logx)
        plt.ylabel('Number of actions')
        plt.xlabel('Time')

        fname = outfname + '_p-{}'.format(p)
        plt.savefig(fname + '.png')
        plt.close()
        df_a.to_csv(fname + '.csv', index=False)

def plot_actions_subcount(df, outfname, actions_filter, ylabel):
    """Plot mean number of times the given actions are taken"""
    df = df[['iteration','policy','a']].copy()
    actions_passing = [a for a in df['a'].dropna().unique() if
                       actions_filter(a)]
    df = df[df.a.isin(actions_passing)]
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

def plot_observations(df, outfname, formatter=None, line=False, logx=True):
    obs = df.groupby(['policy', 't_rel'])['o'].value_counts().unstack().fillna(0.).reset_index()
    if formatter:
        obs.rename(columns=dict((x, formatter(x)) for x in obs.columns[2:]), inplace=True)
    for p, df_o in obs.groupby('policy', as_index=False):
        if not line:
            step_by_col(df_o, 't_rel').plot(x='t_rel', y=obs.columns[2:], kind='area',
                                 title='Observation counts', logx=logx)
        else:
            df_o.plot(x='t_rel', y=obs.columns[2:], kind='line',
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

def plot_n_workers_by_budget(df, outfname):
    df.sort(['t'])
    df['cum_cost'] = -1 * df.groupby(['policy', 'iteration'])['cost'].cumsum()
    df = df.groupby(['policy', 'iteration', 'cum_cost'],
                    as_index=False)['worker'].max()
    ax = tsplot_robust(df, time='cum_cost', condition='policy',
                       unit='iteration', value='worker', ci=CI)

    plt.ylabel('Mean workers hired')
    plt.xlabel('Budget spent')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.savefig(outfname + '.png')
    plt.close()

def plot_n_workers(df, outfname):
    df = df.groupby(['policy', 'iteration'], as_index=False)['worker'].max()
    ax = sns.boxplot(y='policy', x='worker', data=df, orient='h')
    plt.xlabel('Number of workers hired')
    plt.tight_layout()
    plt.savefig(outfname + '.png')
    plt.close()

def plot_timings(df_timings, outfname):
    for t in ('resolve', 'estimate'):
        df_filter = df_timings[df_timings['type'] == t]
        if len(df_filter.index) > 0:
            ax = tsplot_robust(df_filter, time='worker', unit='iteration',
                               condition='policy', value='duration', ci=CI)
            fname = outfname + '-{}'.format(t)
            plt.xlabel('Worker number')
            plt.ylabel('Mean seconds to {}'.format(t))
            plt.savefig(fname + '.png')
            plt.close()

def plot_params(df_model, outfname):
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

def query_names_df(df, s, c, i):
    return df[(df.type == s) & (df.i == int(i))].reset_index()[c][0]

def make_plots(infiles, outdir, models=[], timings=[], names=None,
               policies=None, line=False, log=True, worker_interval=5):
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

    df = pd.concat([pd.read_csv(f) for f in infiles],
                   ignore_index=True)
    if policies is not None:
        df = df[df.policy.isin(policies)]
    
    # New column for 't' relative to worker.
    df.sort('t')
    df['t_rel'] = df.groupby(['policy', 'iteration', 'worker'])['t'].apply(lambda s: s - s.min())

    if len(timings) > 0:
        df_timings = pd.concat([pd.read_csv(f) for f in timings],
                               ignore_index=True)
        if policies is not None:
            df_timings = df_timings[df_timings.policy.isin(policies)]
        if len(df_timings.index) > 0:
            plot_timings(df_timings, os.path.join(outdir, 't'))
            print 'Done plotting timings'

    if action_uses_gold_question is not None:
        plot_actions_subcount(
            df, outfname=os.path.join(outdir, 'gold_questions_used'),
            actions_filter=action_uses_gold_question,
            ylabel='Mean number of gold questions used')

    df_model = pd.concat([pd.read_csv(f) for f in models],
                         ignore_index=True)
    if len(df_model['policy'].dropna().unique()) > 0:
        plot_params(df_model, os.path.join(outdir, 'params'))
    print 'Done plotting params'

    plot_reward_by_t(df, os.path.join(outdir, 'r_t'))
    plot_reward_by_budget(df, os.path.join(outdir, 'r_cost'))
    plot_n_workers_by_budget(df, os.path.join(outdir, 'n_workers_cost'))
    plot_n_workers(df, os.path.join(outdir, 'n_workers'))

    for p, df_filter in df.groupby('policy'):
        last_worker = df_filter['worker'].max()
        worker_indices = range(0, last_worker + 1, worker_interval)
        if worker_indices[-1] != last_worker:
            worker_indices.append(last_worker)
        for w in worker_indices:
            df_worker = df_filter[df_filter.worker == w]
            worker_dir = os.path.join(outdir, 'w', '{:04d}'.format(w))
            util.ensure_dir(worker_dir)
            plot_beliefs(df_worker, os.path.join(worker_dir, 'b'),
                         formatter=str_state, line=line, logx=log)
            plot_actions(df_worker, os.path.join(worker_dir, 'a'),
                         formatter=str_action, line=line, logx=log)
            plot_observations(df_worker, os.path.join(worker_dir, 'o'),
                              formatter=str_observation, line=line, logx=log)

def main(filenames, policies=None, line=False, log=True,
         single=False, dest=None, worker_interval=5):
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
                   worker_interval=worker_interval)
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
                worker_interval=worker_interval))
            jobs.append(p)
            p.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('result', type=str, nargs='+',
                        help='main experiment result .txt files')
    parser.add_argument('--line', dest='line', action='store_true',
                        help="Use line plots instead of area")
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help="Don't use log for x-axis")
    parser.add_argument('--single', dest='single', action='store_true',
                        help='Treat multiple inputs as single experiment')
    parser.add_argument('--dest', '-d', type=str, help='Folder to store plots')
    parser.add_argument('--policies', type=str, nargs='*',
                        help='Policies to use')
    parser.add_argument('--worker_interval', type=int, default=10,
                        help='Interval between plots for worker plots')
    args = parser.parse_args()

    main(filenames=args.result,
         policies=args.policies,
         line=args.line,
         log=args.log,
         single=args.single,
         dest=args.dest,
         worker_interval=args.worker_interval)
