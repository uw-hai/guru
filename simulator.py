"""simulator.py"""

import random
import numpy as np
from pomdp import POMDPModel
import hcomp_data_analyze.load
import work_learn_problem as wlp

COST_WRONG = -1

class Simulator(object):
    """Class for synthetic data."""
    def __init__(self, params):
        n_worker_classes = len(params['p_worker'])
        self.model_gt = POMDPModel(n_worker_classes, params=params)
        self.s = None

    def worker_available(self):
        """Return whether new worker is available."""
        return True

    def new_worker(self):
        """Simulate obtaining a new worker and return start state."""
        start_belief = self.model_gt.get_start_belief()
        start_state = np.random.choice(range(len(start_belief)),
                                       p=start_belief)
        self.s = start_state
        return self.s

    def sample_SOR(self, a):
        """Return (state, observation, (cost, reward)).

        State may be None if the simulator does not provide that.

        Change in false positive cost plus false negative cost.

        """
        s, o, (cost, r) = self.model_gt.sample_SOR(self.s, a)
        # Ignore states that give a new worker from within the model.
        if self.model_gt.observations[o] == 'term':
            s = None
        self.s = s
        self.o = o
        return s, o, (cost, r)

    def worker_hired(self):
        """Return whether worker is currently hired."""
        return self.s is not None

class LiveSimulator(Simulator):
    """Class for live data."""
    def __init__(self, params, dataset):
        self.params = params
        self.observations = wlp.observations
        self.actions = wlp.actions_all(1, tell=False, exp=False)
        if dataset == 'lin_aaai12_tag':
            self.df = hcomp_data_analyze.load.load_lin_aaai12(workflow='tag')
        elif dataset == 'lin_aaai12_wiki':
            self.df = hcomp_data_analyze.load.load_lin_aaai12(workflow='wiki')
        elif dataset == 'rajpal_icml15':
            self.df = hcomp_data_analyze.load.load_rajpal_icml15(
                worker_type=None)
        else:
            raise Exception('Unexpected dataset')
        self.hired = False
        self.workers = list(self.df.groupby('worker'))
        random.shuffle(self.workers)

    def worker_available(self):
        """Return whether new worker is available."""
        return len(self.workers) > 0

    def new_worker(self):
        """Simulate obtaining a new worker and return start state."""
        _, df = self.workers.pop()
        self.worker_o = [self.observations.index('right') if v else
                         self.observations.index('wrong') for
                         v in df['correct'].values]
        random.shuffle(self.worker_o)
        self.hired = True
        return None

    def sample_SOR(self, a):
        """Return (state, observation, (cost, reward)).

        State may be None if the simulator does not provide that.

        Change in false positive cost plus false negative cost.

        """
        if len(self.worker_o) == 0 or self.actions[a].get_type() == 'boot':
            self.o = self.observations.index('term')
            self.hired = False
            return None, self.o, (0, 0)
        else:
            self.o = self.worker_o.pop()

        # Calculate reward.
        if self.actions[a].get_type() == 'work':
            cost = self.params['cost']
            was_right = random.random() <= 0.5
            if self.observations[self.o] == 'right':
                vote_right = True
            elif self.observations[self.o] == 'wrong':
                vote_right = False
            else:
                raise Exception('Unexpected observation')
            cost = self.params['cost']
            if was_right and not vote_right:
                r = COST_WRONG
            elif not was_right and vote_right:
                r = -1 * COST_WRONG
            else:
                r = 0
            self.o = self.observations.index('null')
        elif self.actions[a].get_type() == 'test':
            cost = self.params['cost']
            r = 0
        else:
            raise Exception('Unexpected action')
        return None, self.o, (cost, r)

    def worker_hired(self):
        """Return whether worker is currently hired."""
        return self.hired
