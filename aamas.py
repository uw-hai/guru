import argparse
import os
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

class Plotter(object):
    def __init__(self):
        self.client = self.get_client()
        self.labels = {
            'teach_first-n_tell_0': 'Work-only',
            'test_and_boot-n_test_4-n_work_16-acc_0.7': 'Test-and-boot',
            'test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work': 'Test-and-boot',
            'zmdp-d0.990-tl60': 'POMDP',
            'zmdp-d0.990-tl60-eps_1-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2': 'POMDP-RL',
            'zmdp-d0.990-tl60-eps_1-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-HyperParamsSpacedUnknownRatioSlipLeaveLose-cl2': 'POMDP-RL',
            'zmdp-d0.990-tl60-eps_1-1div(1+e**(-1000*(f-0.5)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2': 'POMDP-RL',
            'zmdp-d0.990-tl60-eps_1-1div(1+e**(-1000*(f-0.5)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-n_blocks_1-final_work-HyperParamsSpacedUnknownRatioSlipLeave-cl2': 'POMDP-RL',
            'zmdp-d0.990-tl60-eps_1-1div(1+e**(-1000*(f-0.5)))-explore_test_work-explore_p_test_and_boot-n_test_4-n_work_16-acc_0.7-HyperParamsSpacedUnknownRatioSlipLeaveLose-cl2': 'POMDP-RL'}

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

    def make_plot(self, policies, experiment, linestyles,
                  markerstyles=None, reserved=False, loc='upper left',
                  markevery=10):
        p = an.ResultPlotter.from_mongo(
            collection=self.client.worklearn.res,
            experiment=experiment,
            policies=policies,
            collection_names=self.client.worklearn.names)
        if reserved:
            p.set_reserved()
        ax = p.plot_reward_by_budget(fill=True, action_cost=0.000001)
        h, l = ax.get_legend_handles_labels()
        d = {self.labels[label]: line for line, label in zip(h, l)}
        labels = [self.labels[p] for p in policies]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fig_n', type=int, choices=[1,2,3,4])
    args = parser.parse_args()

    p = Plotter()
    if args.fig_n == 1:
        p.make_fig1()
    elif args.fig_n == 2:
        p.make_fig2()
    elif args.fig_n == 3:
        p.make_fig3()
    else:
        p.make_fig4()
