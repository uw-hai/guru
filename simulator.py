"""simulator.py"""

import random
import collections
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
        n_skills = len(self.params['p_r'])
        self.n_question_types = len(self.params['p_1'])
        self.observations = wlp.observations(
            n_question_types=self.n_question_types)

        # TODO: Remove dependencies on datasets.
        if dataset['name'] in ['lin_aaai12', 'rajpal_icml15']:
            self.actions = wlp.actions_all(
                n_skills=n_skills, n_question_types=self.n_question_types,
                tell=False, exp=False)
            if self.params['tell'] or self.params['exp']:
                raise ValueError('Unexpected parameter settings')
            if n_skills != 1:
                raise ValueError('Unexpected parameter settings')
            if self.n_question_types > 1:
                raise ValueError('Unexpected parameter settings')
        elif dataset['name'] == 'bragg_teach':
            self.actions = wlp.actions_all(
                n_skills=n_skills, n_question_types=self.n_question_types,
                tell=False, exp=True)
            if self.params['tell'] or not self.params['exp']:
                raise ValueError('Unexpected parameter settings')

        self.df = hanalyze.Data.from_dataset(
            name=dataset['name'], options=dataset['options']).df

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
            """Normalize row from dataframe.

            If row['answer'] is a list or mapping, convert values into
            observation string like 'rrrw' (if first three question types
            are correct).

            """
            d = dict()
            if ('action' not in row or row['action'] == 'ask' and (
                    pd.notnull(row['actiontype']) or
                    self.convert_work_to_quiz)):
                d['a'] = quiz_index
                if row['answertype'] == 'term':
                    o_str = 'term'
                elif (self.n_question_types > 1 and
                        isinstance(row['answer'], collections.Mapping)):
                    o_str = ''.join(
                        'r' if row['answer'][k] == row['gt'][k] else 'w' for
                        k in sorted(row['answer']))
                elif (self.n_question_types > 1 and
                        isinstance(row['answer'], collections.Iterable)):
                    o_str = ''.join(
                        'r' if v1 == v2 else 'w' for
                        v1, v2 in zip(row['answer'], row['gt']))
                else:
                    o_str = 'r' if row['answer'] == row['gt'] else 'w'
                d['o'] = self.observations.index(o_str)
            elif row['action'] == 'ask':
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

            Exception: Unexpected 0-length worker.

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
            metadata = None
        elif self.actions[a].get_type() == 'work':
            metadata = collections.defaultdict(list)
            metadata['ans'] = ans
            cost = self.params['cost']
            penalty_old = []
            penalty_new = []
            reward_old = []
            reward_new = []
            if (self.n_question_types > 1 and
                    isinstance(ans['gt'], collections.Mapping)):
                ans_gt = [ans['gt'][k] for k in sorted(ans['gt'])]
                ans_answer = [ans['answer'][k] for k in sorted(ans['gt'])]
            elif (self.n_question_types > 1 and
                    not isinstance(ans['gt'], collections.Iterable)):
                raise Exception('Multiple question types but not multiple answers.')
            elif self.n_question_types == 1:
                ans_gt = [ans['gt']]
                ans_answer = [ans['answer']]
            for p_1, ans_gt_v, ans_answer_v in zip(
                    self.params['p_1'], ans_gt, ans_answer):
                guess = round(p_1)
                if ans_gt_v == 0 and guess == 1:
                    penalty_old.append(penalty_fp)
                    reward_old.append(0)
                elif ans_gt_v == 1 and guess == 0:
                    penalty_old.append(penalty_fn)
                    reward_old.append(0)
                elif ans_gt_v == 0 and guess == 0:
                    penalty_old.append(0)
                    reward_old.append(reward_tn)
                else:
                    penalty_old.append(0)
                    reward_old.append(reward_tp)

                if self.params['utility_type'] == 'pen_nonboolean':
                    if ans_gt_v == ans_answer_v:
                        reward_new.append(reward_tp)
                        penalty_new.append(0)
                    else:
                        penalty_new.append(penalty_fp)
                        reward_new.append(0)
                elif ans_gt_v == 0 and ans_answer_v == 1:
                    penalty_new.append(penalty_fp)
                    reward_new.append(0)
                elif ans_gt_v == 1 and ans_answer_v == 0:
                    penalty_new.append(penalty_fn)
                    reward_new.append(0)
                elif ans_gt_v == 0 and ans_answer_v == 0:
                    penalty_new.append(0)
                    reward_new.append(reward_tn)
                else:
                    penalty_new.append(0)
                    reward_new.append(reward_tp)

            metadata['answer'] = ans_answer
            metadata['gt'] = ans_gt
            if self.params['utility_type'] in ['pen', 'pen_nonboolean']:
                r_new = np.sum([penalty_new, reward_new], axis=0)
                r = r_new
                metadata['rewards'] = r.tolist()
                r = np.sum(r)
            elif self.params['utility_type'] == 'pen_diff':
                r_new = np.sum([penalty_new, reward_new], axis=0)
                r_old = np.sum([penalty_old, reward_old], axis=0)
                r = r_new - r_old
                metadata['rewards'] = r.tolist()
                r = np.sum(r)
            else:
                raise ValueError('Utility function undefined for live data')

            self.o = self.observations.index('null')
        elif self.actions[a].get_type == 'boot':
            metadata = None
            cost = 0
            r = 0
        elif self.actions[a].get_type() == 'test':
            metadata = None
            cost = self.params['cost']
            r = 0
        elif self.actions[a].get_type() == 'exp':
            metadata = None
            cost = self.params['cost_exp']
            r = 0
        elif self.actions[a].get_type() == 'tell':
            metadata = None
            cost = self.params['cost_tell']
            r = 0
        else:
            raise Exception('Unexpected action {}'.format(self.actions[a]))
        if self.observations[self.o] == 'term':
            self.hired = False
        return a, None, self.o, (cost, r), metadata

    def worker_hired(self):
        """Return whether worker is currently hired."""
        return self.hired
