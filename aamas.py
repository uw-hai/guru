from __future__ import division
import argparse
import os
import csv
import util as ut
import pymongo

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
mpl.rcParams.update({'font.size': 42})
mpl.rc('legend', fontsize=20)
mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)
mpl.rc('axes', labelsize=24)

import analyze as an
import hcomp_data_analyze.analyze as han

class Plotter(object):
    def __init__(self):
        self.client = self.get_client()

        self.linestyles = {
            'POMDP': '-',
            'POMDP-RL': '-',
            'Test-and-boot': '--',
            'Work-only': ':'}

        self.markerstyles = {
            'POMDP-RL': 'o'}


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

    def make_fig1(self):
        policies = ['zmdp-d0.990-tl60',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-20_80_cost_0.000001_pen3_std0.1',
            policies=policies,
            linestyles=self.linestyles)
        self.make_plot(
            experiment='test_classes2-50_50_cost_0.000001_pen3_std0.1',
            policies=policies,
            linestyles=self.linestyles)
        self.make_plot(
            experiment='test_classes2-80_20_cost_0.000001_pen3_std0.1',
            policies=policies,
            linestyles=self.linestyles)

    def make_fig2(self):
        """Plot RL with 50:50, p_lose=0"""
        # Last 90% plot.
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_cost_0.000001_pen3_std0.1',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            reserved=True)
        # Version with epsilon-first 50%.
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-1div(1+e**(-1000*(f-0.5)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_cost_0.000001_pen3_std0.1_rl',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=100,
            reserved=False,
            loc='upper left')

    def make_fig3(self):
        """Plot RL with 50:50, p_lose=0.01"""
        # Last 90% plot.
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
            loc='lower left')
        # Version with epsilon-first 50%.
        policies = ['zmdp-d0.990-tl60',
                    'zmdp-d0.990-tl60-eps_1-1div(1+e**(-1000*(f-0.5)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-HyperParamsSpacedUnknownRatioSlipLeaveLose-cl2',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_lose_0.01_cost_0.000001_pen3_std0.1_rl',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=100,
            reserved=False,
            loc='lower left')

    def make_fig4(self):
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

    def make_fig5(self):
        policies = ['zmdp-d0.990-tl60',
                    'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work',
                    'teach_first-n_tell_0']
        self.make_plot(
            experiment='test_classes2-50_50_cost_0.000001_pen6_std0.1_rl',
            policies=policies,
            linestyles=self.linestyles,
            reserved=False,
            loc='lower left')
        self.make_plot(
            experiment='test_classes2-50_50_cost_0.000001_pen1_std0.1_rl',
            policies=policies,
            linestyles=self.linestyles,
            reserved=False)

    def make_fig6(self):
        """New version of 50:50 learning with different eps"""
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
            experiment='test_classes2-50_50_lose_0.01_cost_0.000001_pen3_std0.1_rl_bud0.004',
            policies=policies,
            linestyles=self.linestyles,
            markerstyles=self.markerstyles,
            markevery=200,
            reserved=False,
            loc='lower left')

    @staticmethod
    def get_name(label):
        """Policy name to nicer version."""
        if label == 'teach_first-n_tell_0':
            return 'Work-only'
        elif label.startswith('test_and_boot'):
            return 'Test-and-boot'
        elif label.startswith('zmdp') and 'eps' in label:
            return 'POMDP-RL'
        elif label.startswith('zmdp'):
            return 'POMDP'
        else:
            raise NotImplementedError

    def make_plot(self, policies, experiment, linestyles,
                  markerstyles=None, reserved=False, loc='upper left',
                  markevery=10):
        """Make a reward vs budget plot."""
        p = an.ResultPlotter.from_mongo(
            collection=self.client.worklearn.res,
            experiment=experiment,
            policies=policies,
            collection_names=self.client.worklearn.names)
        if reserved:
            p.set_reserved()
        ax, sig = p.plot_reward_by_budget(fill=True, action_cost=0.000001)
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
        fname = os.path.join('aamas', experiment)
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
                    dw.writerow({'p1': p1, 'p2': p2,
                                 'tstat': tstat, 'pval': pval})

def make_accuracy_plots():
    for k in ['tag', 'wiki']:
        d = han.Data.from_lin_aaai12(workflow=k)
        ax = d.plot_scatter_n_accuracy()
        ut.savefig(ax, os.path.join('aamas', 'scatter_lin_{}.png'.format(k)))
        plt.close()

    d = han.Data.from_rajpal_icml15(worker_type=None)
    ax = d.plot_scatter_n_accuracy()
    ut.savefig(ax, os.path.join('aamas', 'scatter_rajpal.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fig_n', type=int)
    args = parser.parse_args()

    make_accuracy_plots()

    p = Plotter()
    if args.fig_n == 1:
        p.make_fig1()
    elif args.fig_n == 2:
        p.make_fig2()
    elif args.fig_n == 3:
        p.make_fig3()
    elif args.fig_n == 4:
        p.make_fig4()
    elif args.fig_n == 5:
        p.make_fig5()
    else:
        p.make_fig6()
