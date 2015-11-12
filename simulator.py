"""simulator.py"""

import random
import numpy as np
from pomdp import POMDPModel
import hcomp_data_analyze.analyze
import work_learn_problem as wlp

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
        """Return (state, observation, (cost, reward), other).

        State may be None if the simulator does not provide that.

        Change in false positive cost plus false negative cost.

        """
        s, o, (cost, r) = self.model_gt.sample_SOR(self.s, a)
        # Ignore states that give a new worker from within the model.
        if self.model_gt.observations[o] == 'term':
            s = None
        self.s = s
        self.o = o
        return s, o, (cost, r), None

    def worker_hired(self):
        """Return whether worker is currently hired."""
        return self.s is not None

class LiveSimulator(Simulator):
    """Class for live data."""
    def __init__(self, params, repeat=True):
        """Initialize.

        Args:
            params:     param.Params object.
            repeat:     Allow replay of workers multiple times.

        """
        self.params = params.get_param_dict(sample=False)
        dataset = self.params['dataset']
        self.repeat = repeat
        self.observations = wlp.observations
        n_skills = len(self.params['p_r'])
        if self.params['tell'] or self.params['exp'] or n_skills != 1:
            raise ValueError('Unexpected parameter settings')
        self.actions = wlp.actions_all(n_skills, tell=False, exp=False)
        if dataset == 'lin_aaai12_tag':
            self.df = hcomp_data_analyze.analyze.Data.from_lin_aaai12(
                workflow='tag').df
        elif dataset == 'lin_aaai12_wiki':
            self.df = hcomp_data_analyze.analyze.Data.from_lin_aaai12(
                workflow='wiki').df
        elif dataset == 'rajpal_icml15':
            self.df = hcomp_data_analyze.analyze.Data.from_rajpal_icml15(
                worker_type=None).df
        else:
            raise Exception('Unexpected dataset')
        self.hired = False
        self.init_workers()

    def init_workers(self):
        self.workers = list(self.df.groupby('worker'))
        random.shuffle(self.workers)

    def worker_available(self):
        """Return whether new worker is available."""
        return self.repeat or len(self.workers) > 0

    def new_worker(self):
        """Simulate obtaining a new worker and return start state."""
        if self.repeat and len(self.workers) == 0:
            self.init_workers()
        _, df = self.workers.pop()
        self.worker_ans = [{'o': self.observations.index('right') if
                                 r['correct'] else
                                 self.observations.index('wrong'),
                            'answer': r['answer'],
                            'worker': r['worker'],
                            'question': r['question'],
                            'gt': r['gt']} for _, r in df.iterrows()]
        random.shuffle(self.worker_ans)
        self.hired = True
        return None

    def sample_SOR(self, a):
        """Return (state, observation, (cost, reward), other).

        State may be None if the simulator does not provide that.

        Change in false positive cost plus false negative cost.

        """
        if len(self.worker_ans) == 0 or self.actions[a].get_type() == 'boot':
            self.o = self.observations.index('term')
            self.hired = False
            return None, self.o, (0, 0), None
        else:
            ans = self.worker_ans.pop()
            self.o = ans['o']

        if self.params['utility_type'] == 'acc':
            # BUG: In order to define this, think we need to set
            # reward_correct = 0.5 and penalty = -0.5, but need to verify.
            raise ValueError('Accuracy gain undefined for live data')
        else:
            penalty_fp = self.params['penalty_fp']
            penalty_fn = self.params['penalty_fn']
        # TODO: Get more information from df.

        # Calculate reward.
        if self.actions[a].get_type() == 'work':
            cost = self.params['cost']
            guess = round(self.params['p_1'])
            reward_correct = 1
            if ans['gt'] == 0 and guess == 1:
                penalty_old = penalty_fp
                reward_old = 0
            elif ans['gt'] == 1 and guess == 0:
                penalty_old = penalty_fn
                reward_old = 0
            else:
                penalty_old = 0
                reward_old = reward_correct
            if ans['gt'] == 0 and ans['answer'] == 1:
                penalty_new = penalty_fp
                reward_new = 0
            elif ans['gt'] == 1 and ans['answer'] == 0:
                penalty_new = penalty_fn
                reward_new = 0
            else:
                penalty_new = 0
                reward_new = reward_correct

            if self.params['utility_type'] == 'pen':
                r = penalty_new + reward_new
            elif self.params['utility_type'] == 'pen_diff':
                r = (penalty_new + reward_new) - (penalty_old + reward_old)
            else:
                raise ValueError('Accuracy gain undefined for live data')

            self.o = self.observations.index('null')
        elif self.actions[a].get_type() == 'test':
            cost = self.params['cost']
            r = 0
        else:
            raise Exception('Unexpected action')
        return None, self.o, (cost, r), ans

    def worker_hired(self):
        """Return whether worker is currently hired."""
        return self.hired
