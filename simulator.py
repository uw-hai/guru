"""simulator.py"""

import numpy as np
from pomdp import POMDPModel

class Simulator(object):
    def __init__(self, params):
        n_worker_classes = len(params['p_worker'])
        self.model_gt = POMDPModel(n_worker_classes, params=params)
        self.s = None

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

    def no_worker(self):
        """Return whether current worker has left."""
        return self.s is None
