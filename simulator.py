"""simulator.py"""

import random
import pandas as pd
import numpy as np
from .pomdp import POMDPModel
from .hcomp_data_analyze import analyze as hanalyze
from . import work_learn_problem as wlp

class Simulator(object):
    """Class for synthetic data."""

    def __init__(self, params):
        """Initialize.

        Args:
            params: param.Params object.

        """
        self.params = params
        self.model_gt = None
        self.s = None

    def worker_available(self):
        """Return whether new worker is available."""
        return True

    def new_worker(self):
        """Simulate obtaining a new worker and return start state."""
        self.model_gt = POMDPModel(
            self.params.n_classes,
            params=self.params.get_param_dict(sample=True))

        start_belief = self.model_gt.get_start_belief()
        start_state = np.random.choice(range(len(start_belief)),
                                       p=start_belief)
        self.s = start_state
        return self.s

    def sample_SOR(self, a):
        """Take an action and sample a new state-observation-reward.

        State may be None if the simulator does not provide that.

        Change in false positive cost plus false negative cost.

        Args:
            a (int): Action index.

        Returns:
            (action, state, observation, (cost, reward), other)

        Raises:
            ValueError: Unable to take action because requested action is
                was not taken with worker. 'Boot' action may be taken
                regardless.

        """
        s, o, (cost, r), other = self.model_gt.sample_SOR(self.s, a)
        # Ignore states that give a new worker from within the model.
        if self.model_gt.observations[o] == 'term':
            s = None
        self.s = s
        self.o = o
        return a, s, o, (cost, r), other

    def worker_hired(self):
        """Return whether worker is currently hired."""
        return self.s is not None


class LiveSimulator(Simulator):
    """Class for live data."""

    def __init__(self, params, repeat=False,
                 random_workers=False, random_actions=False,
                 convert_work_to_quiz=False):
        """Initialize.

        Assumes one skill.

        Args:
            params (.param.Params): Params object.
                params['dataset'] must be one of:
                - 'lin_aaai12_tag'
                - 'lin_aaai12_wiki'
                - 'rajpal_icml15'
                - 'bragg_teach_pilot_3_20'
                - 'bragg_teach_rl_v2'
            repeat (bool): Allow replay of workers multiple times.
            random_workers (bool): Randomize worker order.
            random_actions (bool): Randomize action order within worker.
            convert_work_to_quiz (bool): Treat all work actions as quiz.

        Raises:
            ValueError: If parameters don't match required parameters
                for dataset.

        """
        self.random_workers = random_workers
        self.random_actions = random_actions
        self.convert_work_to_quiz = convert_work_to_quiz
        self.params = params.get_param_dict(sample=False)
        dataset = self.params['dataset']
        self.repeat = repeat
        self.observations = wlp.observations
        n_skills = len(self.params['p_r'])

        if n_skills != 1:
            raise ValueError('Unexpected parameter settings')
        if dataset in ['lin_aaai12_tag', 'lin_aaai12_wiki', 'rajpal_icml15']:
            self.actions = wlp.actions_all(n_skills, tell=False, exp=False)
            if self.params['tell'] or self.params['exp']:
                raise ValueError('Unexpected parameter settings')
        elif dataset.startswith('bragg_teach'):
            self.actions = wlp.actions_all(n_skills, tell=False, exp=True)
            if self.params['tell'] or not self.params['exp']:
                raise ValueError('Unexpected parameter settings')

        if dataset == 'lin_aaai12_tag':
            self.df = hanalyze.Data.from_lin_aaai12(
                workflow='tag').df
        elif dataset == 'lin_aaai12_wiki':
            self.df = hanalyze.Data.from_lin_aaai12(
                workflow='wiki').df
        elif dataset == 'rajpal_icml15':
            self.df = hanalyze.Data.from_rajpal_icml15(
                worker_type=None).df
        elif dataset == 'bragg_teach_pilot_3_20':
            self.df = hanalyze.Data.from_bragg_teach(
                conditions=['pilot_3', 'pilot_20']).df
        elif dataset == 'bragg_teach_rl_v2':
            self.df = hanalyze.Data.from_bragg_teach(
                conditions=['rl_v2']).df
        else:
            raise Exception('Unexpected dataset')
        # Reverse chronological order for .pop() to work.
        self.df = self.df.sort('time', ascending=False)
        self.hired = False
        self.init_workers()

    def init_workers(self):
        self.workers = list(self.df.groupby('worker'))
        if self.random_workers:
            random.shuffle(self.workers)

    def worker_available(self):
        """Return whether new worker is available."""
        return self.repeat or len(self.workers) > 0

    def new_worker(self):
        """Simulate obtaining a new worker and return start state.

        Assume all actions are quiz actions if action not provided.

        Raises:
            ValueError: Unexpected worker data.

        """
        work_index = self.actions.index(wlp.Action('ask', None))
        quiz_index = self.actions.index(wlp.Action('ask', 0))
        if self.params['exp']:
            exp_index = self.actions.index(wlp.Action('exp'))
        if self.repeat and len(self.workers) == 0:
            self.init_workers()
        _, df = self.workers.pop()
        def normalize_row(row):
            """Normalize row from dataframe."""
            d = dict()
            if 'action' not in row:
                d['a'] = quiz_index
                if row['correct']:
                    d['o'] = self.observations.index('right')
                else:
                    d['o'] = self.observations.index('wrong')
            elif row['action'] == 'ask':
                if pd.notnull(row['actiontype']) or self.convert_work_to_quiz:
                    d['a'] = quiz_index
                    if row['correct']:
                        d['o'] = self.observations.index('right')
                    else:
                        d['o'] = self.observations.index('wrong')
                else:
                    d['a'] = work_index
                    d['o'] = self.observations.index('null')
            elif row['action'] == 'exp':
                d['a'] = exp_index
                d['o'] = self.observations.index('null')
            else:
                raise ValueError('Unexpected worker data: {}'.format(row))
            d['answer'] = row['answer']
            d['worker'] = row['worker']
            d['question'] = row['question']
            d['gt'] = row['gt']
            return d

        self.worker_ans = [normalize_row(row) for _, row in df.iterrows()]
        if self.random_actions:
            random.shuffle(self.worker_ans)
        # Ensure every sequence of actions ends with a term observation.
        term_index = self.observations.index('term')
        if self.worker_ans[0]['o'] != term_index:
            self.worker_ans[0]['o'] = term_index
        self.hired = True

    def sample_SOR(self, a=None):
        """Take an action and sample a new state-observation-reward.

        State may be None if the simulator does not provide that.

        Change in false positive cost plus false negative cost.

        Args:
            a (int): Action index. Defaults to next action in sequence.

        Returns:
            (action, state, observation, (cost, reward), other)

        Raises:
            ValueError: Unable to take action because requested action is
                was not taken with worker. 'Boot' action may be taken
                regardless. Also raised if accuracy gain reward is requested
                for live data.

            Exception: Unexpected 0-length wo

        """
        if not self.hired:
            raise Exception("Unable to sample when worker not hired.")
        elif len(self.worker_ans) == 0:
            raise ValueError('Unexpected end of worker history.')
        else:
            ans = self.worker_ans.pop()
            if a is None:
                a = ans['a']
            # TODO: Allow taking "work" action when "quiz" action was recorded.
            elif a != ans['a'] and self.actions[a].get_type() != 'boot':
                raise ValueError('Requested action index {} when {} is next action'.format(a, ans['a']))
            if self.actions[a].get_type() == 'boot':
                self.o = self.observations.index('term')
            else:
                self.o = ans['o']

        if self.params['utility_type'] == 'acc':
            # BUG: In order to define this, think we need to set
            # reward_correct = 0.5 and penalty = -0.5, but need to verify.
            raise ValueError('Accuracy gain undefined for live data')
        else:
            penalty_fp = self.params['penalty_fp']
            penalty_fn = self.params['penalty_fn']
            reward_tp = self.params['reward_tp']
            reward_tn = self.params['reward_tn']
        # TODO: Get more information from df.

        # Calculate reward.
        if self.observations[self.o] == 'term':
            cost = 0
            r = 0
        elif self.actions[a].get_type() == 'work':
            cost = self.params['cost']
            guess = round(self.params['p_1'])
            if ans['gt'] == 0 and guess == 1:
                penalty_old = penalty_fp
                reward_old = 0
            elif ans['gt'] == 1 and guess == 0:
                penalty_old = penalty_fn
                reward_old = 0
            elif ans['gt'] == 0 and guess == 0:
                penalty_old = 0
                reward_old = reward_tn
            else:
                penalty_old = 0
                reward_old = reward_tp

            if ans['gt'] == 0 and ans['answer'] == 1:
                penalty_new = penalty_fp
                reward_new = 0
            elif ans['gt'] == 1 and ans['answer'] == 0:
                penalty_new = penalty_fn
                reward_new = 0
            elif ans['gt'] == 0 and ans['answer'] == 0:
                penalty_new = 0
                reward_new = reward_tn
            else:
                penalty_new = 0
                reward_new = reward_tp

            if self.params['utility_type'] == 'pen':
                r = penalty_new + reward_new
            elif self.params['utility_type'] == 'pen_diff':
                r = (penalty_new + reward_new) - (penalty_old + reward_old)
            else:
                raise ValueError('Accuracy gain undefined for live data')

            self.o = self.observations.index('null')
        elif self.actions[a].get_type == 'boot':
            cost = 0
            r = 0
        elif self.actions[a].get_type() == 'test':
            cost = self.params['cost']
            r = 0
        elif self.actions[a].get_type() == 'exp':
            cost = self.params['cost_exp']
            r = 0
        elif self.actions[a].get_type() == 'tell':
            cost = self.params['cost_tell']
            r = 0
        else:
            raise Exception('Unexpected action {}'.format(self.actions[a]))
        if self.observations[self.o] == 'term':
            self.hired = False
        return a, None, self.o, (cost, r), ans

    def worker_hired(self):
        """Return whether worker is currently hired."""
        return self.hired
