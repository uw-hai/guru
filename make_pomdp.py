"""make_pomdp.py"""

from __future__ import division
import json
from util import get_or_default
import numpy as np
import work_learn_problem as wlp
from work_learn_problem import Action

class POMDPWriter:
    """Generator for .pomdp files"""
    def __init__(self, **params):
        self.params = params
        self.n_skills = len(self.params['p_s'])
        self.n_states_base = 2 ** self.n_skills
        self.actions = wlp.actions(self.n_skills)
        self.states = wlp.states_all(self.n_skills)
        self.observations = wlp.observations

        self.p_t, self.p_o, self.rewards = self.make_tables()

    def write(self, fo, discount):
        """Write a Cassandra-style POMDP spec with the given discount"""
        if discount >= 1.0:
            raise Exception('Discount must be less than 1.0')

        # Write header
        fo.write('discount: {}\n'.format(discount))
        fo.write('values: reward\n')
        fo.write('states: {}\n'.format(' '.join(str(s) for s in self.states)))
        fo.write('actions: {}\n'.format(' '.join(str(a) for a in self.actions)))
        fo.write('observations: {}\n'.format(' '.join(self.observations)))

        fo.write('start: {}\n'.format(' '.join(
            str(x) for x in self.get_start_belief())))

        fo.write('\n\n### Transitions\n')
        for s,st in enumerate(self.states):
            for a,act in enumerate(self.actions):
                for s1,st1 in enumerate(self.states):
                    fo.write('T: {} : {} : {} {}\n'.format(
                        act, st, st1, self.p_t[s][a][s1]))
                fo.write('\n')

        fo.write('\n\n### Observations\n')
        for s,st in enumerate(self.states):
            for a,act in enumerate(self.actions):
                for o,obs in enumerate(self.observations):
                    fo.write('O: {} : {} : {} {}\n'.format(
                        act, st, obs, self.p_o[s][a][o]))
                fo.write('\n')

        fo.write('\n\n### Rewards\n')
        for s,st in enumerate(self.states):
            for a,act in enumerate(self.actions):
                for s1,st1 in enumerate(self.states):
                    fo.write('R: {} : {} : {} : * {}\n'.format(
                        act, st, st1, self.rewards[s][a][s1]))
                fo.write('\n')

    def get_start_belief(self):
        return [self.start_probability(s) for s in self.states]

    def start_probability(self, state):
        """Get start probability of a state"""
        if state.term or state.is_quiz():
            return 0
        else:
            prob = 1
            for v, p in zip(state.skills, self.params['p_s']):
                if v:
                    prob *= p
                else:
                    prob *= 1 - p
            return prob

    def make_tables(self):
        """Helper function for __init__ that creates model tables
        
        Returns:
            p_t (|S|.|A|.|S| array):        Transition probabilties
            p_o (|S|.|A|.|O| array):        Observation probabilities
            rewards (|S|.|A|.|S| array):    Rewards
        """
        p_s = self.params['p_s']
        p_r = self.params['p_r']
        p_slip = self.params['p_slip']
        p_guess = self.params['p_guess']
        p_1 = self.params['p_1']
        cost = self.params['cost']
        cost_exp = self.params['cost_exp']
        p_leave = self.params['p_leave']
        p_lose = get_or_default(self.params, 'p_lose', 0)
        utility_type = get_or_default(self.params, 'utility_type', 'acc')
        p_learn = self.params['p_learn']

        # Initialize tables
        p_t = np.zeros((len(self.states), len(self.actions), len(self.states)))
        p_o = np.zeros((len(self.states), len(self.actions), len(self.observations)))
        rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))

        
        # Initialize identity transformations.
        for s,_ in enumerate(self.states):
            for a,_ in enumerate(self.actions):
                p_t[s][a][s] = 1
        # Observations equally likely, except term observation.
        for s,_ in enumerate(self.states):
            for a,_ in enumerate(self.actions):
                for o,obs in enumerate(self.observations):
                    if obs != 'term':
                        p_o[s][a][o] = 1 / (len(self.observations) - 1)

        # Set transitions and rewards.
        for s,st in enumerate(self.states):
            for a,act in enumerate(self.actions):
                if not st.is_valid_action(act):
                    rewards[s][a][s] = wlp.NINF
            if st.term:
                continue
            for s1,st1 in enumerate(self.states):
                if not st.is_quiz() and st1.term:
                    # Executed once for each root starting stats.

                    # Booting takes to terminal state.
                    p_t[s][self.actions.index(Action('boot'))][s1] = 1
                    p_t[s][self.actions.index(Action('boot'))][s] = 0
                    # Worker might leave when we quiz or ask
                    for a,act in enumerate(self.actions):
                        if act.is_quiz() or act.name == 'ask':
                            p_t[s][a][s1] = p_leave
                            p_t[s][a][s] = 0
                elif st1.term:
                    # Done with terminal state.
                    # Important: Disallow booting from quiz states.
                    continue
                elif not st.is_quiz() and not st1.is_quiz():
                    if st.is_reachable(st1, False):
                        n_known = st.n_skills_known()
                        n_lost = st.n_skills_lost(st1)
                        n_lost_not = n_known - n_lost
                        prob = 1 * p_lose ** n_lost * (1 - p_lose) ** n_lost_not
                        prob *= 1 - p_leave
                        p_t[s][self.actions.index(Action('ask'))][s1] = prob
                        rewards[s][self.actions.index(Action('ask'))][s1] += \
                            cost
                        rewards[s][self.actions.index(Action('ask'))][s1] += \
                            st.rewards_ask(p_r, p_slip, p_guess, p_1,
                                           utility_type)
                elif (not st.is_quiz() and st1.is_quiz() and
                      st.has_same_skills(st1)):
                    # Action to get to st1.
                    a = self.actions.index(Action('quiz', st1.quiz_val))
                    p_t[s][a][s1] = 1 - p_leave
                    rewards[s][a][s1] += cost
                elif st.is_quiz() and not st1.is_quiz():
                    if st.is_reachable(st1, False):
                        n_known = st.n_skills_known()
                        n_lost = st.n_skills_lost(st1)
                        n_lost_not = n_known - n_lost
                        prob = 1 * p_lose ** n_lost * \
                               (1 - p_lose) ** n_lost_not
                        a = self.actions.index(Action('noexp'))
                        p_t[s][a][s1] = prob
                        p_t[s][a][s] = 0
                    if st.is_reachable(st1, True):
                        quiz_skill_known = st.has_skill(st.quiz_val)
                        n_known = st.n_skills_known()
                        n_lost = st.n_skills_lost(st1)
                        n_lost_not = n_known - n_lost
                        if (quiz_skill_known):
                            # Can't lose the quiz skill.
                            n_lost_not -= 1
                        prob = 1 * p_lose ** n_lost * \
                               (1 - p_lose) ** n_lost_not
                        if st.n_skills_learned(st1) == 1:
                            prob *= p_learn
                        elif not quiz_skill_known:
                            # Missed opportunity.
                            prob *= 1 - p_learn
                        a = self.actions.index(Action('exp'))
                        p_t[s][a][s1] = prob
                        p_t[s][a][s] = 0
                        rewards[s][a][s1] += cost_exp

        # Set observations
        for s,st in enumerate(self.states):
            if st.term:
               # Always know when we enter terminal state.
               for a,act in enumerate(self.actions):
                    p_o[s][a][self.observations.index('right')] = 0
                    p_o[s][a][self.observations.index('wrong')] = 0
                    p_o[s][a][self.observations.index('term')] = 1
            elif st.is_quiz():
               # Assume teaching actions ask questions that require only the
               # skill being taught.
               p_r_gold = [int(i == st.quiz_val) for i in xrange(self.n_skills)]
               
               has_skills = st.p_has_skills(p_r_gold) == 1
               pr = st.p_right(p_r_gold, p_slip, p_guess)
               a = self.actions.index(Action('quiz', st.quiz_val))
               p_o[s][a][self.observations.index('right')] = pr
               p_o[s][a][self.observations.index('wrong')] = 1 - pr
               p_o[s][a][self.observations.index('term')] = 0

        return p_t, p_o, rewards 
