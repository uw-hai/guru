from __future__ import division
import argparse
import os
import csv
import pymongo
import scipy.stats as ss
import pandas as pd

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt

from . import util as ut
from . import analyze as an
from .hcomp_data_analyze import analyze as han

mpl.rcParams.update({'font.size': 42})
mpl.rc('legend', fontsize=20)
mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)
mpl.rc('axes', labelsize=24)


class Plotter(object):
    def __init__(self):
        self.client = self.get_client()

        self.linestyles = {
            'POMDP-oracle': '-',
            'POMDP-RL': '-',
            'Test-and-boot': '--',
            'Test-and-boot-once': '--',
            'Work-only': ':'}

        self.markerstyles = {
            'POMDP-RL': 'o'}


        self.experiments = {
            'test_classes2-20_80_lose_0.01_cost_0.000001_pen3_std0.1_rl_bud0.004': 'ratio_20_80',
            'test_classes2-50_50_lose_0.01_cost_0.000001_pen3_std0.1_rl_bud0.004': 'ratio_50_50',
            'test_fixed_classes2-50_50_lose_0.01_cost_0.000001_pen3_std0.1': 'ratio_50_50_zoom',
            'test_classes2-80_20_lose_0.01_cost_0.000001_pen3_std0.1_rl_bud0.004': 'ratio_80_20',
            'test_classes2-50_50_cost_0.000001_pen3_std0.1_rl_bud0.004': 'p_lose_0',
            'test_classes2-50_50_lose_0.02_cost_0.000001_pen3_std0.1_rl_bud0.004': 'p_lose_0.02',
            'test_classes2-50_50_lose_0.04_cost_0.000001_pen3_std0.1_rl_bud0.004': 'p_lose_0.04',
            'test_classes2-50_50_lose_0.01_cost_0.000001_pen1_std0.1_rl_bud0.004': 'reward_pen1',
            'test_classes2-50_50_lose_0.01_cost_0.000001_pen6_std0.1_rl_bud0.004': 'reward_pen6',
            'test_lin_aaai12_tag_cost_0.000001_pen3': 'live_lin_tag',
            'test_lin_aaai12_wiki_cost_0.000001_pen3': 'live_lin_wiki',
            'test_rajpal_icml15_cost_0.000001_pen3': 'live_rajpal',
            'test_lin_aaai12_tag_cost_0.000001_pen3_rl': 'live_lin_tag_rl',
            'test_lin_aaai12_wiki_cost_0.000001_pen3_rl': 'live_lin_wiki_rl',
            'test_rajpal_icml15_cost_0.000001_pen3_rl': 'live_rajpal_rl',
            'test_lin_aaai12_tag_cost_0.000001_pen3_rl_0.85': 'live_lin_tag_rl_0.85',
            'test_lin_aaai12_wiki_cost_0.000001_pen3_rl_0.85': 'live_lin_wiki_rl_0.85',
            'test_rajpal_icml15_cost_0.000001_pen3_rl_0.85': 'live_rajpal_rl_0.85',
            }


    def gather_stats(self):
        dfs = []
        for e in self.experiments:
            try:
                f = e
                if ('aaai12' in e or 'icml15' in e) and not 'rl' in  e:
                    f += '_reserved'
                df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'aamas', '{}_acc.csv'.format(f)))
                df2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'aamas', '{}_mean.csv'.format(f)))
                df = df1.merge(df2, left_on='policy', right_on='p')
                df['experiment_full'] = e
                df['experiment'] = self.experiments[e]
                dfs.append(df)
            except:
                pass
        df = pd.concat(dfs, axis=0).sort(['experiment', 'policy'])
        df.to_csv(os.path.join(os.path.dirname(__file__), 'aamas', 'summary.csv'), index=False,
                  columns=['experiment', 'policy', 'mean', 'diff', 'n', 'correct', 'accuracy', 'n_err', 'correct_err', 'accuracy_err'])

    @staticmethod
    def get_client():
        mongo={
           'host': os.environ['MONGO_HOST'],
           'port': int(os.environ['MONGO_PORT']),
           'user': ut.get_or_default(os.environ, 'MONGO_USER', None),
           'pass': ut.get_or_default(os.environ, 'MONGO_PASS', None),
           'db': os.environ['MONGO_DBNAME']}

        client = pymongo.MongoClient(mongo['host'], mongo['port'])
        if mongo['user']:
            client[mongo['db']].authenticate(mongo['user'], mongo['pass'],
                                          mechanism='SCRAM-SHA-1')
        return client

    def make_vary_ratio(self, new=True):
        """Vary ratio, p_lose = 0.01"""
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-1div(1+e**(-40*(f-0.4)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-HyperParamsSpacedUnknownRatioSlipLeaveLose-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_lose_0.01_cost_0.000001_pen3_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            loc='lower left')
        self.make_plot(
            experiment='test_classes2-20_80_lose_0.01_cost_0.000001_pen3_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            loc='lower left')
        self.make_plot(
            experiment='test_classes2-80_20_lose_0.01_cost_0.000001_pen3_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            loc='lower left')

    def make_reserved_plots(self):
        """Last 10% plots"""
        # 50:50, p_lose = 0, budget = 0.002
        """
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_cost_0.000001_pen3_std0.1',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            reserved=True,
            loc='upper left')
        """

        # 50:50, p_lose = 0.01, budget = 0.002
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-HyperParamsSpacedUnknownRatioSlipLeaveLose-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_fixed_classes2-50_50_lose_0.01_cost_0.000001_pen3_std0.1',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            reserved=True,
            loc='lower left',
            xlim=[1800, 2000])

    def make_vary_p_lose(self):
        """Vary p_lose with 50:50"""
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-1div(1+e**(-40*(f-0.4)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_cost_0.000001_pen3_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            reserved=False,
            loc='upper left')
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-1div(1+e**(-40*(f-0.4)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-HyperParamsSpacedUnknownRatioSlipLeaveLose-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_lose_0.02_cost_0.000001_pen3_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            reserved=False,
            loc='lower left')
        self.make_plot(
            experiment='test_classes2-50_50_lose_0.04_cost_0.000001_pen3_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            reserved=False,
            loc='lower left')

    def make_live(self):
        policies = ['zmdp-d0.990-tl60-eps_1-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_lin_aaai12_tag_cost_0.000001_pen3',
            policies=policies,
            linestyles=self.linestyles,
            reserved=True)
        self.make_plot(
            experiment='test_lin_aaai12_wiki_cost_0.000001_pen3',
            policies=policies,
            linestyles=self.linestyles,
            reserved=True)
        self.make_plot(
            experiment='test_rajpal_icml15_cost_0.000001_pen3',
            policies=policies,
            linestyles=self.linestyles,
            reserved=True)

    def make_live2(self):
        policies = ['zmdp-d0.990-tl60-eps_1-1div(1+e**(-40*(f-0.4)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_lin_aaai12_tag_cost_0.000001_pen3_rl',
            policies=policies,
            linestyles=self.linestyles)
        self.make_plot(
            experiment='test_lin_aaai12_wiki_cost_0.000001_pen3_rl',
            policies=policies,
            linestyles=self.linestyles)
        self.make_plot(
            experiment='test_rajpal_icml15_cost_0.000001_pen3_rl',
            policies=policies,
            linestyles=self.linestyles)

    def make_live3(self):
        policies = ['zmdp-d0.990-tl60-eps_1ifwlt20else0-explore_test_work-explore_p_test_and_boot-n_test_7-n_work_16-acc_0.85-n_blocks_1-final_work-HyperParamsUnknownRatioLeave-cl2-acc0.85',
                    'test_and_boot-n_test_7-n_work_16-acc_0.85-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_lin_aaai12_tag_cost_0.000001_pen3_rl_0.85',
            policies=policies,
            linestyles=self.linestyles,
            loc='lower left')
        self.make_plot(
            experiment='test_lin_aaai12_wiki_cost_0.000001_pen3_rl_0.85',
            policies=policies,
            linestyles=self.linestyles)
        self.make_plot(
            experiment='test_rajpal_icml15_cost_0.000001_pen3_rl_0.85',
            policies=policies,
            linestyles=self.linestyles)


    def make_vary_reward(self):
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-1div(1+e**(-40*(f-0.4)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-HyperParamsSpacedUnknownRatioSlipLeaveLose-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_lose_0.01_cost_0.000001_pen6_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            reserved=False,
            loc='lower left')
        self.make_plot(
            experiment='test_classes2-50_50_lose_0.01_cost_0.000001_pen1_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            reserved=False)

    @staticmethod
    def get_name(label):
        """Policy name to nicer version."""
        if label == 'teach_first-n_tell_0':
            return 'Work-only'
        elif label.startswith('test_and_boot') and 'n_blocks' in label:
            return 'Test-and-boot-once'
        elif label.startswith('test_and_boot'):
            return 'Test-and-boot'
        elif label.startswith('zmdp') and 'eps' in label and 'UnknownRatioLeave' in label:
            #return 'POMDP-RL-binned'
            return 'POMDP-RL'
        elif label.startswith('zmdp') and 'eps' in label:
            return 'POMDP-RL'
        elif label.startswith('zmdp'):
            return 'POMDP-oracle'
        else:
            raise NotImplementedError

    def find_accuracies(self, policies, experiment, result_plotter=None,
                        reserved=False):
        """Make dataset size and accuracy"""
        if result_plotter:
            rp = result_plotter
        else:
            rp = an.ResultPlotter.from_mongo(
                collection=self.client.worklearn.res,
                experiment=experiment,
                policies=policies,
                collection_names=self.client.worklearn.names)
        if reserved:
            rp.set_reserved()
        df = rp.get_work_stats()
        if df is None:
            return None

        df['accuracy'] = df['correct'] / df['n']
        means = df.groupby(['policy']).mean()
        errors = 1.96 * df.groupby(['policy']).agg(ss.sem)
        means['n_err'] = errors['n']
        means['correct_err'] = errors['correct']
        means['accuracy_err'] = errors['accuracy']
        means = means.reset_index(drop=False)
        return means.replace(dict((k, self.get_name(k)) for k in
                                  means['policy'].unique()))

    def make_plot(self, policies, experiment, linestyles,
                  markerstyles=None, reserved=False, loc='upper left',
                  markevery=10, xlim=None):
        """Make a reward vs budget plot."""
        rp = an.ResultPlotter.from_mongo(
            collection=self.client.worklearn.res,
            experiment=experiment,
            policies=policies,
            collection_names=self.client.worklearn.names)
        ax, sig, means = rp.plot_reward_by_budget(fill=True,
                                                  action_cost=0.000001,
                                                  reserved=reserved)
        if xlim is not None:
            ax.set_xlim(*xlim)
        h, l = ax.get_legend_handles_labels()
        d = {self.get_name(label): line for line, label in zip(h, l)}
        labels = [self.get_name(p) for p in policies]
        for k in d:
            if linestyles and k in linestyles:
                d[k].set_linestyle(linestyles[k])
            if markerstyles and k in markerstyles:
                d[k].set_marker(markerstyles[k])
                d[k].set_markevery(markevery)
        ax.legend([d[l] for l in labels], labels, loc=loc)
        ax.set_xlabel('Number of questions asked')
        fname = os.path.join(os.path.dirname(__file__), 'aamas', experiment)
        if reserved:
            fname += '_reserved'

        ut.savefig(ax, '{}.png'.format(fname))
        plt.close()
        if sig is not None:
            with open('{}_sig.csv'.format(fname), 'w') as f:
                dw = csv.DictWriter(f, ['p1', 'p2', 'tstat', 'pval'])
                dw.writeheader()
                for k in sig:
                    p1, p2 = k
                    tstat, pval = sig[k]
                    dw.writerow({'p1': self.get_name(p1),
                                 'p2': self.get_name(p2),
                                 'tstat': tstat, 'pval': pval})
            with open('{}_mean.csv'.format(fname), 'w') as f:
                worst = min(means.itervalues())
                dw = csv.DictWriter(f, ['p', 'mean', 'diff'])
                dw.writeheader()
                for k in means:
                    dw.writerow({
                        'p': self.get_name(k),
                        'mean': means[k],
                        'diff': means[k] - worst})
        df = self.find_accuracies(policies, experiment, result_plotter=rp)
        if df is not None:
            df.to_csv('{}_acc.csv'.format(fname), index=False)

def make_accuracy_plots():
    for k in ['tag', 'wiki']:
        d = han.Data.from_lin_aaai12(workflow=k)
        ax = d.plot_scatter_n_accuracy()
        ut.savefig(ax, os.path.join(os.path.dirname(__file__), 'aamas', 'scatter_lin_{}.png'.format(k)))
        plt.close()

    d = han.Data.from_rajpal_icml15(worker_type=None)
    ax = d.plot_scatter_n_accuracy()
    ut.savefig(ax, os.path.join(os.path.dirname(__file__), 'aamas', 'scatter_rajpal.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fig_n', type=int)
    args = parser.parse_args()

    #make_accuracy_plots()

    p = Plotter()
    p.gather_stats()
    if args.fig_n == 1:
        p.make_vary_ratio()
    elif args.fig_n == 2:
        p.make_vary_p_lose()
    elif args.fig_n == 3:
        p.make_reserved_plots()
    elif args.fig_n == 4:
        p.make_vary_reward()
    elif args.fig_n == 5:
        p.make_live()
    elif args.fig_n == 6:
        p.make_live2()
    else:
        p.make_live3()
    p.gather_stats()
