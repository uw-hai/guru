"""analyze.py

Analyze work_learn output.

TODO: Handle new model params format.
TODO: Parse model 'hyper' column, in place of alpha and beta.

"""

import multiprocessing as mp
import time
import os
import shutil
import csv
import ast
import argparse
import collections
import itertools
import logging
import functools as ft
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
import util
from util import savefig, tsplot_robust
import work_learn_problem as wlp
import pymongo

CI = 95  # Confidence interval

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)


def rename_classes_h(d):
    """Move ModelPlotter classmethod to top-level for MultiProcessing."""
    return ModelPlotter.rename_classes_h(**d)

class Plotter(object):
    def __init__(self, df, df_names=None):
        self.df = df
        self.df_full = df
        self.df_names = df_names
        self.uses_gold_known = (df_names is not None and
                                'uses_gold' in df_names.columns)

    def restore(self):
        self.df = self.df_full

    def set_quantile(self, quantile=[0, 1]):
        self.df = self.filter_workers_quantile(self.df_full, *quantile)

    def set_reserved(self):
        if 'reserved' in self.df_full:
            self.df = self.df_full[self.df_full.reserved]

    @classmethod
    def from_mongo(cls, collection, experiment, policies=None,
                   collection_names=None):
        """Return dataframe of rows from given Mongo collection.

        Args:
            collection: Pymongo collection object.
            collection: Pymongo collection object for names information.

        """
        if not policies:
            policies = collection.find({
                'experiment': experiment}).distinct('policy')
        results = collection.find(
            {'experiment': experiment,
             'policy': {'$in': policies}},
            {'_id': False,
             'experiment': False})
        df = pd.DataFrame(list(results))
        if len(df) == 0:
            raise ValueError('No rows')

        if collection_names is not None:
            results_names = collection_names.find(
                {'experiment': experiment},
                {'_id': False})
            df_names = pd.DataFrame(list(results_names))
            if len(df_names) == 0:
                return cls(df)
            return cls(df, df_names)
        else:
            return cls(df)

    @staticmethod
    def filter_workers_quantile(df, q1, q2):
        """Filter dataframe for iterations with number of workers in given range.

        Args:
            df:     Dataframe
            q1:     Lower quantile
            q2:     Upper quantile

        Returns:
            Dataframe.

        """
        max_workers = df.groupby(['policy', 'iteration'])['worker'].max()
        quantiles = max_workers.quantile([q1, q2])

        max_workers_in_range = max_workers[(max_workers >= quantiles[q1]) &
                                           (max_workers <= quantiles[q2])]

        cols = df.columns.values
        joined = df.join(max_workers_in_range,
                         on=['policy', 'iteration'], how='inner',
                         rsuffix='_max')
        return joined[cols]

    @staticmethod
    def step_by_col(df, col, interval=0.1):
        """Return a new dataframe with data copied in step function manner.

        Copies point at t=i to the range t=[i, i+1) at the given interval.
        Area plots look jagged, so this makes the lines more vertical.

        """
        dfs = []
        for i in np.arange(0, 1, interval):
            df2 = df.copy()
            df2[col] = df2[col].map(lambda x: x + i)
            dfs.append(df2)
        df_out = pd.concat(dfs, ignore_index=True)
        df_out.sort(['policy', col], inplace=True)
        #df_out.reset_index()
        return df_out

    def action_uses_gold(self, i):
        """Return whether the given action uses a gold question."""
        return self._query_names_df(self.df_names, 'action', 'uses_gold', i)

    def get_name(self, type_, i):
        """Get name associated with index."""
        if self.df_names is not None:
            return self._query_names_df(self.df_names, type_, 's', i)
        else:
            return str(i)

    def _query_names_df(self, df, s, c, i):
        return df[(df.type == s) & (df.i == int(i))].reset_index()[c][0]

class JoinPlotter(Plotter):
    """Plotter for join of three data files."""
    def __init__(self, df, df_model, df_timings, df_names=None):
        super(JoinPlotter, self).__init__(df, df_names)
        self.model_plotter = ModelPlotter(df_model)
        self.df_timings = df_timings

    def get_traces(self, policy):
        df = self.df[self.df.policy == policy]
        df = df.sort(['iteration', 'worker'])
        for (it, w), df_i in df.groupby(['iteration', 'worker']):
            timings = self.df_timings[(self.df_timings.iteration == it) &
                                      (self.df_timings.worker == w)]
            model = self.model_plotter.df[(self.model_plotter.df.iteration == it) &
                                          (self.model_plotter.df.worker == w)]
            yield df_i.sort('t'), timings, model


class ResultPlotter(Plotter):
    """Plotter for main result file."""
    def __init__(self, df, df_names=None):
        super(ResultPlotter, self).__init__(df, df_names)
        # New column for 't' relative to worker.
        self.df = self.df.sort('t')
        self.df['t_rel'] = self.df.groupby(
            ['policy', 'iteration', 'worker'])['t'].apply(lambda s: s - s.min())
        self.df_full = self.df

    def make_plots(self, outdir, worker_interval, line, logx):
        if self.uses_gold_known:
            self.plot_actions_subcount(
                outfname=os.path.join(outdir, 'gold_questions_used'),
                ylabel='Mean number of gold questions used')

        quart123dir = os.path.join(outdir, 'quart123')
        quart4dir = os.path.join(outdir, 'quart4')
        reserveddir = os.path.join(outdir, 'reserved')
        util.ensure_dir(quart123dir)
        util.ensure_dir(quart4dir)
        util.ensure_dir(reserveddir)

        def f(self, d):
            print 'making', os.path.join(d, 't.png')
            ax = self.plot_iteration_runtime()
            savefig(ax, os.path.join(d, 't.png'))
            plt.close()
            self.plot_actions_by_worker_agg(
                os.path.join(d, 'n_actions_by_worker'))
            self.plot_actions_by_worker(
                os.path.join(d, 'n_actions_by_worker'))
            self.plot_reward_by_t(os.path.join(d, 'r_t'))
            self.plot_reward_by_budget(os.path.join(d, 'r_cost'))
            self.plot_n_workers_by_budget(os.path.join(d, 'n_workers_cost'))

        for d, q in [(outdir, [0, 1]),
                     (quart123dir, [0, 0.75]),
                     (quart4dir, [0.75, 1])]:
            self.set_quantile(q)
            f(self, d)
        self.set_reserved()
        f(self, reserveddir)
        self.restore()
        self.plot_n_workers(os.path.join(outdir, 'n_workers'))

        for p, df_filter in self.df.groupby('policy'):
            last_worker = df_filter['worker'].max()
            worker_indices = range(0, last_worker + 1, worker_interval)
            if worker_indices[-1] != last_worker:
                worker_indices.append(last_worker)
            for w in worker_indices:
                df_worker = df_filter[df_filter.worker == w]
                worker_dir = os.path.join(outdir, 'w', '{:04d}'.format(w))
                util.ensure_dir(worker_dir)
                self.plot_beliefs(
                    df_worker, os.path.join(worker_dir, 'b'),
                    line=line, logx=logx)
                self.plot_actions(
                    df_worker, os.path.join(worker_dir, 'a'),
                    line=line, logx=logx)
                self.plot_observations(
                    df_worker, os.path.join(worker_dir, 'o'),
                    line=line, logx=logx)

    def plot_beliefs(self, df, outfname, line=False, logx=True):
        """Plot beliefs for subset of entire dataframe"""
        df_b = pd.DataFrame(df.b.str.split().tolist()).astype(float)
        states = df_b.columns

        df = df.join(df_b)
        b_sums = df.groupby(['policy','t_rel'], as_index=False).sum()
        b_sums.rename(columns=dict((x, self.get_name('state', x)) for
                                   x in states),
                      inplace=True)
        states = [self.get_name('state', x) for x in df_b.columns]
        for p, df_b in b_sums.groupby('policy', as_index=False):
            if not line:
                self.step_by_col(df_b, 't_rel').plot(
                    x='t_rel', y=states, kind='area',
                    title='Belief counts', logx=logx)
            else:
                df_b.plot(x='t_rel', y=states, kind='line',
                          title='Belief counts', logx=logx)
            fname = outfname + '_p-{}'.format(p)
            ax = plt.gca()
            savefig(ax, fname + '.png')
            plt.close()
            df_b.to_csv(fname + '.csv', index=False)

    def plot_actions_by_worker_agg(self, outfname, logx=False, logy=True):
        """Plot timeseries with mean number of actions per worker."""
        df = self.df
        actions = df.groupby(
            ['policy', 'iteration', 'worker'], as_index=False)['t_rel'].max()
        ax, df_stat = tsplot_robust(actions, time='worker', condition='policy',
                                    unit='iteration', value='t_rel', ci=95,
                                    logx=logx, logy=logy)
        plt.ylim(0, None)
        plt.xlim(0, None)
        plt.xlabel('Worker')
        plt.ylabel('Mean number of actions')
        savefig(ax, outfname + '.png')
        plt.close()
        df_stat.to_csv(outfname + '.csv', index=False)

    def plot_actions_by_worker(self, outfname, line=True,
                               logx=False, logy=True):
        df = self.df
        actions = df.groupby(['policy', 'iteration', 'worker'])['a'].value_counts().unstack().fillna(0.).reset_index()
        actions.rename(columns=dict((x, self.get_name('action', x)) for
                                    x in actions.columns[3:]), inplace=True)
        actions = actions.groupby(['policy', 'worker'], as_index=False)[actions.columns.values[3:]].mean()
        for p, df_a in actions.groupby('policy', as_index=False):
            if not line:
                self.step_by_col(df_a, 'worker').plot(
                    x='worker', y=sorted(actions.columns[2:]),
                    kind='area', logx=logx)
            else:
                if logy:
                    df_a[actions.columns[2:]] += 1
                df_a.plot(x='worker', y=sorted(actions.columns[2:]),
                          kind='line', logx=logx, logy=logy)
            plt.xlim(0, None)
            plt.ylim(0, None)
            ylabel = 'Mean number of actions'
            if logy:
                ylabel += ' (plus 1)'
            plt.ylabel(ylabel)
            plt.xlabel('Worker')

            fname = outfname + '_p-{}'.format(p)
            ax = plt.gca()
            savefig(ax, fname + '.png')
            plt.close()
            df_a.to_csv(fname + '.csv', index=False)

    def plot_actions(self, df, outfname, line=True, logx=False):
        """Plot actions for subset of entire dataframe"""
        actions = df.groupby(['policy', 't_rel'])['a'].value_counts().unstack().fillna(0.).reset_index()
        actions.rename(columns=dict((x, self.get_name('action', x)) for
                                    x in actions.columns[2:]), inplace=True)
        for p, df_a in actions.groupby('policy', as_index=False):
            if not line:
                self.step_by_col(df_a, 't_rel').plot(
                    x='t_rel', y=sorted(actions.columns[2:]),
                    kind='area', logx=logx)
            else:
                df_a.plot(x='t_rel', y=sorted(actions.columns[2:]),
                          kind='line', logx=logx)
            plt.ylabel('Number of actions')
            plt.xlabel('Time')

            fname = outfname + '_p-{}'.format(p)
            ax = plt.gca()
            savefig(ax, fname + '.png')
            plt.close()
            df_a.to_csv(fname + '.csv', index=False)

    def plot_actions_subcount(self, outfname, ylabel):
        """Plot mean number of times the given actions are taken"""
        df = self.df
        df = df[['iteration','policy','a']].copy()
        actions_passing = [a for a in df['a'].dropna().unique() if
                           self.action_uses_gold(a)]
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
        savefig(ax, outfname + '.png')
        plt.close()

    def plot_observations(self, df, outfname, line=False, logx=True):
        """Plot observations for subset of entire dataframe"""
        obs = df.groupby(['policy', 't_rel'])['o'].value_counts().unstack().fillna(0.).reset_index()
        obs.rename(columns=dict((x, self.get_name('observation', x)) for
                                x in obs.columns[2:]), inplace=True)
        for p, df_o in obs.groupby('policy', as_index=False):
            if not line:
                self.step_by_col(df_o, 't_rel').plot(x='t_rel', y=obs.columns[2:], kind='area',
                                     title='Observation counts', logx=logx)
            else:
                df_o.plot(x='t_rel', y=obs.columns[2:], kind='line',
                          title='Observation counts', logx=logx)
            fname = outfname + '_p-{}'.format(p)
            ax = plt.gca()
            savefig(ax, fname + '.png')
            plt.close()
            df_o.to_csv(fname + '.csv', index=False)

    def plot_reward_by_t(self, outfname):
        df = self.df
        #df = df[df.t <= min(df.groupby(['policy','iteration'])['t'].max())]
        #print df.groupby(['policy','iteration'])['t'].max()
        #print df.groupby(['policy','iteration'])['t'].min()
        #print df.groupby(['policy', 'iteration']).count()
        #print '!!!!!'
        df['reward'] = df['r'].fillna(0) + df['cost'].fillna(0)
        df['csr'] = df.groupby(['policy','iteration'])['reward'].cumsum()
        #ax = sns.tsplot(df, time='t', condition='policy', unit='iteration', value='csr', ci=CI)

        ax = df.groupby(['policy','t'], as_index=False)['csr'].mean().pivot(index='t', columns='policy', values='csr').plot()
        plt.ylabel('Cumulative reward')
        plt.xlabel('t')
        ax.set_xlim(0, None)
        savefig(ax, outfname + '.png')
        plt.close()
        #df.sort(['policy','iteration','episode']).to_csv(fname + '.csv',
        #                                                 index=False)

    def plot_reward_by_budget(self, outfname):
        df = self.df
        df['r'] = df['r'].fillna(0)
        df['cum_r'] = df.groupby(['policy', 'iteration'])['r'].cumsum()
        df['cum_cost'] = -1 * df.groupby(['policy', 'iteration'])['cost'].cumsum()
        df = df.groupby(['policy', 'iteration', 'cum_cost'],
                        as_index=False)['cum_r'].last()
        #ax = df.groupby(['policy', 'cum_cost'], as_index=False)['cum_r'].mean().pivot(index='cum_cost', columns='policy', values='cum_r').plot()
        ax, _ = tsplot_robust(df, time='cum_cost', condition='policy',
                              unit='iteration', value='cum_r', ci=CI)

        plt.ylabel('Cumulative reward')
        plt.xlabel('Budget spent')
        ax.set_xlim(0, None)
        savefig(ax, outfname + '.png')
        plt.close()

    def plot_n_workers_by_budget(self, outfname):
        df = self.df
        df['cum_cost'] = -1 * df.groupby(['policy', 'iteration'])['cost'].cumsum()
        df = df.groupby(['policy', 'iteration', 'cum_cost'],
                        as_index=False)['worker'].max()
        ax, _ = tsplot_robust(df, time='cum_cost', condition='policy',
                              unit='iteration', value='worker', ci=CI)

        plt.ylabel('Mean workers hired')
        plt.xlabel('Budget spent')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        savefig(ax, outfname + '.png')
        plt.close()

    def plot_n_workers(self, outfname):
        df = self.df
        df = df.groupby(['policy', 'iteration'], as_index=False)['worker'].max()
        ax = sns.boxplot(y='policy', x='worker', data=df, orient='h')
        plt.xlabel('Number of workers hired')
        savefig(ax, outfname + '.png')
        plt.close()

    def plot_iteration_runtime(self):
        df = self.df
        df = df.groupby(['policy', 'iteration'], as_index=False)['sys_t'].agg(
                lambda x: (x.max() - x.min()) / 3600)
        ax = sns.boxplot(y='policy', x='sys_t', data=df, orient='h')
        plt.xlabel('Iteration time (hours)')
        return ax


class TimingsPlotter(Plotter):
    def __init__(self, df):
        super(TimingsPlotter, self).__init__(df)
        if any(df['worker'].isnull()):
            raise ValueError('Null workers')

    def make_plots(self, outfname):
        df_timings = self.df
        for t in ('resolve', 'estimate'):
            df_filter = df_timings[df_timings['type'] == t]
            if len(df_filter) > 0:
                ax, _ = tsplot_robust(df_filter,
                                      time='worker', unit='iteration',
                                      condition='policy', value='duration',
                                      ci=CI)
                fname = outfname + '-{}'.format(t)
                plt.xlabel('Worker number')
                plt.ylabel('Mean seconds to {}'.format(t))
                savefig(ax, fname + '.png')
                plt.close()

class ModelPlotter(Plotter):
    def __init__(self, df):
        super(ModelPlotter, self).__init__(df)
        # Ensure params are strings for merge.
        df['param'] = df['param'].astype(str)
        # Separate ground truth params
        df_gt = self.df[df.iteration.isnull()]
        df_est = self.df[df.iteration.notnull()]

        if len(df_gt) > 0:
            # Find dist from true param.
            df_est = df_est.merge(df_gt, how='left', on='param',
                                  suffixes=('', '_t'))

            df_est['dist'] = np.subtract(
                df_est['v'].apply(lambda x: np.array(x)[:-1]),
                df_est['v_t'].apply(lambda x: np.array(x)[:-1]))
            df_est['dist_l1'] = df_est['dist'].apply(
                lambda x: np.linalg.norm(x, 1))
            df_est['dist_l2'] = df_est['dist'].apply(
                lambda x: np.linalg.norm(x, 2))

        if np.all(df_est.hyper.notnull()):
            df_est['dirichlet_mean'] = df_est['hyper'].apply(ss.dirichlet.mean)
            df_est['dirichlet_var_l1'] = df_est['hyper'].apply(
                lambda x: np.linalg.norm(ss.dirichlet.var(x), 1))
            df_est['dirichlet_mode'] = df_est['hyper'].apply(util.dirichlet_mode)

        # Split up dirichlet_mean, dirichlet_mode, and v columns for plotting
        cols = []
        for c in ['v', 'dirichlet_mean', 'dirichlet_mode']:
            if c in df_est.columns:
                s = df_est[c].apply(lambda x: x[:-1]).apply(pd.Series, 1).stack()
                s.name = '{}_single'.format(c)
                cols.append(s)
        df_split = pd.concat(cols, axis=1, keys=[s.name for s in cols])
        df_split = df_split.reset_index(level=1)
        df_split.rename(columns={'level_1': 'ind'}, inplace=True)
        df_split = df_est[['policy', 'iteration', 'worker', 'param']].join(df_split)
        df_split['param'] = df_split['param'] + '-' + df_split['ind'].astype(str)

        # Store.
        self.df_gt = df_gt
        self.df_est = df_est
        self.df_split = df_split

        self.df_est_full = self.df_est
        self.df_split_full = self.df_split

    def set_quantile(self, quantile=[0, 1]):
        super(ModelPlotter, self).set_quantile(quantile)
        self.df_est = self.filter_workers_quantile(self.df_est_full, *quantile)
        self.df_split = self.filter_workers_quantile(self.df_split_full,
                                                     *quantile)

    def get_param_component_traces(self, param, policy):
        df = self.df_split[(self.df_split.param == param) &
                           (self.df_split.policy == policy)]
        for i, df_i in df.groupby('iteration'):
            yield df_i.sort('worker')['v_single']

    @classmethod
    def from_mongo(cls, collection, experiment, policies=None, processes=None):
        """Return dataframe of rows from given Mongo collection.

        Does alignment preprocessing for model.

        Args:
            collection: Pymongo collection object.
            processes:  Number of processes to use.

        """
        # Load, copied from from_mongo in super().
        if not policies:
            policies = collection.find({
                'experiment': experiment,
                'policy': {'$ne': None}}).distinct('policy')
        results = collection.find(
            {'experiment': experiment,
             'policy': {'$in': policies}},
            {'_id': False,
             'experiment': False})
        df = pd.DataFrame(list(results))
        if len(df) == 0:
            raise ValueError('No rows')

        # Preprocessing.
        df_est = df
        df_gt = pd.DataFrame(list(
            collection.find(
                {'experiment': experiment,
                 'policy': None},
                {'_id': False,
                 'experiment': False})))

        if len(df_est) == 0:
            raise ValueError('No estimated rows')

        # Preprocess as needed.
        if len(df_gt) > 0 and 'param_aligned' not in df_est:
            # Separate ground truth params
            df_est_aligned = cls.rename_classes(df_gt, df_est, processes)
            df_est['v'] = df_est_aligned['v']
            df_est['param'] = df_est_aligned['param']
        elif len(df_gt) == 0:
            print 'Empty model gt dataframe'
        else:
            print 'Model already aligned'

        # Initialize
        df = pd.concat([df_est, df_gt], ignore_index=True)
        if len(df['policy'].dropna().unique()) > 0:
            return cls(df)
        else:
            raise ValueError

    @classmethod
    def rename_classes(cls, df_gt, df_est, processes=None):
        """Return version of df_est with classes aligned to closest in gt"""
        df_gt = df_gt.sort('param')
        df_est = df_est.sort('param')

        nprocesses = processes or util.cpu_count()
        pool = mp.Pool(processes=nprocesses,
                       initializer=util.init_worker)
        lst = [{'df_gt': df_gt, 'df_est': df} for _, df in
               df_est.groupby(['iteration', 'policy', 'worker'])]
        f = rename_classes_h
        df_est_renamed = []
        for res in pool.imap_unordered(f, lst):
            df_est_renamed.append(res)
        return pd.concat(df_est_renamed)

    @classmethod
    def rename_classes_h(cls, df_gt, df_est):
        """Helper for inner loop of rename_classes."""
        v_gt = cls.df_params_to_vec(df_gt)
        v_est = cls.df_params_to_vec(df_est)
        m = cls.best_matching(v_est, v_gt)
        params = df_est['param'].map(
            lambda x: x if x == 'p_worker' else ast.literal_eval(x))
        params_renamed = params.map(
            lambda p: p if (not isinstance(p, tuple) or
                            len(p) == 2 and p[1] is None) else \
                      tuple([p[0], m[p[1]]]))
        df_est['param'] = params_renamed
        class_ratio = df_est[df_est.param == 'p_worker']
        class_ratio['v'] = class_ratio['v'].map(
            lambda x: [x[i] for i in [m[k] for k in sorted(m)]])
        return pd.concat([df_est[df_est.param != 'p_worker'], class_ratio])

    @staticmethod
    def best_matching(v1, v2):
        """Return mapping with lowest average distance between classes."""
        assert len(v1) == len(v2)
        c1 = list(v1)
        best_c2 = None
        best_dist = float('inf')
        for c2 in itertools.permutations(v2):
            vec1 = np.vstack([v1[c] for c in c1])
            vec2 = np.vstack([v2[c] for c in c2])
            dist = np.subtract(vec1, vec2)
            dist_l1 = np.linalg.norm(dist, ord=1, axis=1)
            dist_ave = np.mean(dist_l1)
            if dist_ave < best_dist:
                best_dist = dist_ave
                best_c2 = c2
        return dict(zip(c1, best_c2))

    @staticmethod
    def df_params_to_vec(df):
        """Return dictionary from class to parameter vector"""
        params = df['param'].map(lambda x: x if x == 'p_worker' else ast.literal_eval(x))
        params_conditional = params[params.map(lambda x: isinstance(x, tuple))]
        classes = params_conditional.map(lambda x: x[1])
        classes.name = 'class'

        class_ratio = df[df['param'] == 'p_worker']['v'].iloc[0]
        df = pd.concat([df, classes], axis=1)
        res = dict()
        for c, df_c in df.groupby('class'):
            c = int(c)
            res[c] = np.hstack([class_ratio[c], np.hstack(df_c['v'].map(lambda x: x[:-1]))])
        return res

    def make_plots(self, outdir):
        quart123dir = os.path.join(outdir, 'quart123')
        quart4dir = os.path.join(outdir, 'quart4')
        util.ensure_dir(quart123dir)
        util.ensure_dir(quart4dir)

        for d, q in [(outdir, [0, 1]),
                     (quart123dir, [0, 0.75]),
                     (quart4dir, [0.75, 1])]:
            self.set_quantile(q)
            self.plot_params_vector(os.path.join(outdir, 'params'))
            self.plot_params_component(os.path.join(outdir, 'params'))
        self.set_quantile([0, 1])

    def plot_params_vector(self, outfname):
        """Plot param vector properties."""
        df_est = self.df_est
        df_est['param-iteration'] = df_est['param'] + '-' + df_est['iteration'].astype(str)
        for s in ['l1', 'l2']:
            if 'dist_{}'.format(s) in df_est:
                ax, _ = tsplot_robust(
                    df_est, time='worker', unit='param-iteration',
                    condition='policy', value='dist_{}'.format(s),
                    ci=CI)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, None)
                plt.ylabel(
                    'Mean distance ({}) from true parameter values'.format(s))
                plt.xlabel('Worker')
                fname = outfname + '_dist_{}_mean'.format(s)
                savefig(ax, fname + '.png')
                plt.close()

        for p, df_p in df_est.groupby('policy', as_index=False):
            for s in ['dist_l1', 'dist_l2', 'dirichlet_var_l1']:
                if s in df_p:
                    ax, df_stat = tsplot_robust(
                            df_p, time='worker', unit='iteration',
                            condition='param', value=s, ci=CI)
                    ax.set_ylim(0, 1)
                    ax.set_xlim(0, None)
                    if s.startswith('dist'):
                        plt.ylabel('{} from true parameter value'.format(s))
                    else:
                        plt.ylabel('Posterior {}'.format(s))
                    plt.xlabel('Worker')
                    fname = outfname + '_{}_p-{}'.format(s, p)
                    savefig(ax, fname + '.png')
                    plt.close()
                    df_stat.to_csv(fname + '.csv', index=False)

    def plot_params_component(self, outfname):
        """Plot param component properties."""
        df_split = self.df_split
        for p, df_p in df_split.groupby('policy', as_index=False):
            for s in ['v', 'dirichlet_mean', 'dirichlet_mode']:
                if '{}_single'.format(s) in df_p.columns:
                    ax, df_stat = tsplot_robust(
                            df_p, time='worker',
                            unit='iteration', condition='param',
                            value='{}_single'.format(s), ci=CI)
                    ax.set_ylim(0, 1)
                    ax.set_xlim(0, None)
                    plt.ylabel('Estimated parameter {}'.format(s))
                    plt.xlabel('Worker')
                    fname = outfname + '_{}_p-{}'.format(s, p)
                    savefig(ax, fname + '.png')
                    plt.close()
                    df_stat.to_csv(fname + '.csv', index=False)


def run_function_from_dictionary(f, d):
    """Helper for Pool.map(), which can only use functions that take a single
    argument"""
    return f(**d)

def make_plots_h(**kwargs):
    client = pymongo.MongoClient(os.environ['MONGO_HOST'],
                                 int(os.environ['MONGO_PORT'])) 
    client.admin.authenticate(os.environ['MONGO_USER'],
                              os.environ['MONGO_PASS'])
    return make_plots(db=client.worklearn, **kwargs)
 

def make_plots(db, experiment, outdir=None, policies=None,
               line=False, log=True, worker_interval=5,
               processes=None):
    """Make plots for a single experiment.

    Args:
        db:         Pymongo database connection.
        log:        Use log scale for breakdown plots
        policies:   List of policies to include
        processes:  Number of processes to use.

    """
    if outdir is None:
        outdir = os.path.join('static', 'plots', experiment)
    util.ensure_dir(outdir)

    #df = pd.concat([pd.read_csv(f) for f in infiles],
    #               ignore_index=True)
    #if policies is not None:
    #    df = df[df.policy.isin(policies)]
    
    #if len(timings) > 0:
    #    df_timings = pd.concat([pd.read_csv(f) for f in timings],
    #                           ignore_index=True)
    #    if policies is not None:
    #        df_timings = df_timings[df_timings.policy.isin(policies)]
    #    if len(df_timings.index) > 0:
    #        tplotter = TimingsPlotter(df_timings)
    #        tplotter.make_plots(os.path.join(outdir, 't'))
    try:
        tplotter = TimingsPlotter.from_mongo(
            collection=db.timing, experiment=experiment, policies=policies)
        tplotter.make_plots(os.path.join(outdir, 't'))
        print 'Done plotting timings'
    except ValueError:
        pass

    try:
        mplotter = ModelPlotter.from_mongo(
            collection=db.model, experiment=experiment, policies=policies,
            processes=processes)
        print 'Made model plotter'
        mplotter.make_plots(outdir)
        print 'Done plotting params'
    except ValueError:
        pass

    try:
        rplotter = ResultPlotter.from_mongo(
            collection=db.res, experiment=experiment, policies=policies,
            collection_names=db.names)
        rplotter.make_plots(outdir, worker_interval=worker_interval, line=line,
                            logx=log)
        print 'Done plotting result'
    except ValueError:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize policies.')
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--proc', type=int, help='Number of processes')
    parser.add_argument('--line', dest='line', action='store_true',
                        help="Use line plots instead of area")
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help="Don't use log for x-axis")
    parser.add_argument('--dest', '-d', type=str,
                        help='Folder to store plots')
    parser.add_argument('--policies', type=str, nargs='*',
                        help='Policies to use')
    parser.add_argument('--worker_interval', type=int, default=10,
                        help='Interval between plots for worker plots')
    args = parser.parse_args()

    client = pymongo.MongoClient(os.environ['MONGO_HOST'],
                                 int(os.environ['MONGO_PORT'])) 
    client.admin.authenticate(os.environ['MONGO_USER'],
                              os.environ['MONGO_PASS'])

    experiments = list(client.worklearn.res.distinct('experiment'))
    if args.experiment:
        if args.dest is None:
            args.dest = os.path.join('static', 'plots', args.experiment)
        make_plots(
            db=client.worklearn,
            experiment=args.experiment,
            outdir=args.dest,
            policies=args.policies,
            line=args.line,
            log=args.log,
            worker_interval=args.worker_interval,
            processes=args.proc)
    else:
        if args.dest is None:
            args.dest = os.path.join('static', 'plots')
        nprocesses = args.proc or util.cpu_count()
        pool = mp.Pool(processes=nprocesses,
                       initializer=util.init_worker)
        f = ft.partial(util.run_functor,
                       ft.partial(run_function_from_dictionary,
                                  make_plots_h))

        experiments = list(client.worklearn.res.distinct('experiment'))
        args = [{'experiment': e,
                 'outdir': os.path.join(args.dest, e),
                 'policies': None,
                 'line': args.line,
                 'log': args.log,
                 'worker_interval': args.worker_interval} for e in experiments]
        try:
            pool.map(f, args)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            logger.warn('Control-C pressed')
            pool.terminate()
        finally:
            pass
     
