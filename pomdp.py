"""pomdp.py"""

from __future__ import division
import csv
import util
from util import get_or_default
import numpy as np
from numpy import log
import random
from scipy.misc import logsumexp
from scipy.optimize import minimize, check_grad
import work_learn_problem as wlp
from work_learn_problem import Action

import elementtree.ElementTree as ee

class POMDPModel:
    """POMDP model"""
    def __init__(self, **params):
        self.n_skills = len(params['p_r'])
        self.actions = wlp.actions(self.n_skills)
        self.states = wlp.states_all(self.n_skills)
        self.observations = wlp.observations
        self.single_params = [
            'p_guess', 'p_slip', 'p_lose', 'p_learn', 'p_leave']
        self.params = params

        self.beta_priors = {
            'p_s': [(1.1,1.1) for i in xrange(self.n_skills)],
            'p_guess': (1.1, 1.1),
            'p_slip': (2, 5), # Lower probability of making a mistake.
            'p_lose': (2, 5), # Lower probability of forgetting.
            'p_learn': (1.1, 1.1),
            'p_leave': (1.1, 1.1)}

    def write_params(self, fo, writeheader=True, iteration=None, episode=None, policy=None):
        """DEPRECATED: Write params to model csv file"""
        fieldnames = ['iteration', 'episode', 'policy', 'param', 'v']
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        if writeheader:
            writer.writeheader()
        rows = self.get_params_est()
        for r in rows:
            writer.writerow(r)

    def get_params_est(self, iteration=None, episode=None, policy=None):
        """Get estimated parameters in row format to write to model csv"""
        rows = []
        row_base = {'iteration': iteration,
                    'episode': episode,
                    'policy': str(policy)}
        for i in xrange(self.n_skills):
            row = {'param': 'p_s{}'.format(i), 'v': self.params['p_s'][i]}
            row.update(row_base)
            rows.append(row)
        for p in self.single_params:
            row = {'param': p, 'v': self.params[p]}
            row.update(row_base)
            rows.append(row)
        return rows

    def write_names(self, fo):
        """Write csv file mapping state/action/observation indices to names"""
        writer = csv.writer(fo)
        writer.writerow(['i','type','s'])
        for i,a in enumerate(self.actions):
            writer.writerow([i, 'action', a])
        for i,s in enumerate(self.states):
            writer.writerow([i, 'state', s])
        for i,o in enumerate(self.observations):
            writer.writerow([i, 'observation', o])

    def write_txt(self, fo):
        """Write model to file as needed for AI-Toolbox."""
        for s,_ in enumerate(self.states):
            for a,_ in enumerate(self.actions):
                for s1,_ in enumerate(self.states):
                    fo.write('{}\t{}\t'.format(self.get_transition(s, a, s1),
                                               self.get_reward(s, a, s1)))
            fo.write('\n')
        for s,_ in enumerate(self.states):
            for a,_ in enumerate(self.actions):
                for o,_ in enumerate(self.observations):
                    fo.write('{}\t'.format(self.get_observation(s, a, o)))
            fo.write('\n')
                                           

    def write_pomdp(self, fo, discount):
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
                        act, st, st1, self.get_transition(s, a, s1)))
                fo.write('\n')

        fo.write('\n\n### Observations\n')
        for s,st in enumerate(self.states):
            for a,act in enumerate(self.actions):
                for o,obs in enumerate(self.observations):
                    fo.write('O: {} : {} : {} {}\n'.format(
                        act, st, obs, self.get_observation(s, a, o)))
                fo.write('\n')

        fo.write('\n\n### Rewards\n')
        for s,st in enumerate(self.states):
            for a,act in enumerate(self.actions):
                for s1,st1 in enumerate(self.states):
                    fo.write('R: {} : {} : {} : * {}\n'.format(
                        act, st, st1, self.get_reward(s, a, s1)))
                fo.write('\n')

    def get_start_belief(self, params=None):
        if params is None:
            params = self.params
        return [self.get_start_probability(s, params) for
                s in xrange(len(self.states))]

    def get_start_probability(self, s, params=None, derivative=None):
        """Get start probability, or
        derivative with respect to a parameter.

        Args:
            s:          State index
            params:
            derivative: Skill number

        """
        if params is None:
            params = self.params
        st = self.states[s]
        if st.term or st.is_quiz():
            return 0
        else:
            prob = 1
            dprob = 1
            for i, (v, p) in enumerate(zip(st.skills, params['p_s'])):
                if v:
                    prob *= p
                    if derivative != i:
                        dprob *= p 
                else:
                    prob *= 1 - p
                    if derivative != i:
                        dprob *= 1 - p 
                    else:
                        dprob *= -1
            if derivative is None:
                return prob
            else:
                return dprob

    def get_transition(self, s, a, s1, params=None, derivative=None):
        """Get transition probability, or derivative

        Args:
            s:          State index (starting)
            a:          Action index
            s2:         State index (ending)
            params:
            derivative: Can be None, 'p_slip', 'p_guess', 'p_leave',
                        'p_lose', 'p_learn'

        """
        if params is None:
            params = self.params
        p_slip = params['p_slip']
        p_guess = params['p_guess']
        p_leave = params['p_leave']
        p_lose = params['p_lose']
        p_learn = params['p_learn']

        st = self.states[s]
        act = self.actions[a]
        st1 = self.states[s1]

        def cval(v):
            """Derivative of constant values is always 0"""
            if derivative is None:
                return v
            else:
                return 0

        # Actions from terminal state.
        if st.term and st1.term:
            return cval(1)
        elif st.term:
            return cval(0)
        
        # Actions from non-quiz states.
        if not st.is_quiz() and act.name == 'boot':
            # Booting takes to terminal state.
            if st1.term:
                return cval(1)
            else:
                return cval(0)
        elif not st.is_quiz() and act.name == 'ask':
            if st1.term:
                if derivative is None:
                    return p_leave
                elif derivative == 'p_leave':
                    return 1
                else:
                    return 0
            elif st1.is_quiz() or not st.is_reachable(st1, False):
                return cval(0)
            else:
                n_known = st.n_skills_known()
                n_lost = st.n_skills_lost(st1)
                n_lost_not = n_known - n_lost
                prob = p_lose ** n_lost * (1 - p_lose) ** n_lost_not
                if derivative is None:
                    return prob * (1 - p_leave)
                elif derivative == 'p_leave':
                    return -1 * prob
                elif derivative == 'p_lose':
                    # f g'
                    term1 = p_lose ** n_lost * \
                            n_lost_not * (1 - p_lose) ** (n_lost_not - 1) * -1
                    # f' g
                    term2 = n_lost * p_lose ** (n_lost - 1) * \
                            (1 - p_lose) ** n_lost_not
                    return (term1 + term2) * (1 - p_leave)
                else:
                    return 0
        elif not st.is_quiz() and act.is_quiz():
            if st1.term:
                if derivative is None:
                    return p_leave
                elif derivative == 'p_leave':
                    return 1
                else:
                    return 0
            elif (st1.is_quiz() and st.has_same_skills(st1) and
                  act.quiz_val == st1.quiz_val):
                if derivative is None:
                    return 1 - p_leave
                elif derivative == 'p_leave':
                    return -1
                else:
                    return 0
            else:
                return cval(0)
        elif not st.is_quiz():
            if s == s1:
                return cval(1)
            else:
                return cval(0)

        # Actions from quiz states.
        if st.is_quiz() and act.name == 'neoxp':
            if (not st1.term and not st1.is_quiz() and
                    st.is_reachable(st1, False)):
                n_known = st.n_skills_known()
                n_lost = st.n_skills_lost(st1)
                n_lost_not = n_known - n_lost
                prob = p_lose ** n_lost * (1 - p_lose) ** n_lost_not
                if derivative is None:
                    return prob
                elif derivative == 'p_lose':
                    # f g'
                    term1 = p_lose ** n_lost * \
                            n_lost_not * (1 - p_lose) ** (n_lost_not - 1) * -1
                    # f' g
                    term2 = n_lost * p_lose ** (n_lost - 1) * \
                            (1 - p_lose) ** n_lost_not
                    return term1 + term2
                else:
                    return 0
            else:
                return cval(0)
        elif st.is_quiz() and act.name == 'exp':
            if (not st1.term and not st1.is_quiz() and
                    st.is_reachable(st1, True)):
                quiz_skill_known = st.has_skill(st.quiz_val)
                n_known = st.n_skills_known()
                n_lost = st.n_skills_lost(st1)
                n_lost_not = n_known - n_lost
                if (quiz_skill_known):
                    # Can't lose the quiz skill.
                    n_lost_not -= 1
                prob = p_lose ** n_lost * (1 - p_lose) ** n_lost_not
                # f g'
                if p_lose == 0 or p_lose == 1:
                    # TODO: Fix hack
                    dprob = 0
                else:
                    dprob = p_lose ** n_lost * \
                            n_lost_not * (1 - p_lose) ** (n_lost_not - 1) * -1
                    # f' g
                    dprob += n_lost * p_lose ** (n_lost - 1) * \
                             (1 - p_lose) ** n_lost_not
                if st.n_skills_learned(st1) == 1:
                    if derivative is None:
                        return prob * p_learn
                    elif derivative == 'p_learn':
                        return prob
                    elif derivative == 'p_lose':
                        return dprob * p_learn
                    else:
                        return 0
                elif not quiz_skill_known:
                    # Missed opportunity.
                    if derivative is None:
                        return prob * (1 - p_learn)
                    elif derivative == 'p_learn':
                        return prob * -1
                    elif derivative == 'p_lose':
                        return dprob * (1 - p_learn)
                    else:
                        return 0
                else:
                    if derivative is None:
                        return prob
                    elif derivative == 'p_lose':
                        return dprob
                    else:
                        return 0
            else:
                return cval(0)
        else:
            if s == s1:
                return cval(1)
            else:
                return cval(0)

    def get_reward(self, s, a, s1, params=None):
        """Get reward

        Args:
            s:          State index (starting)
            a:          Action index
            s2:         State index (ending)
            params:

        """
        if params is None:
            params = self.params
        p_r = params['p_r']
        p_slip = params['p_slip']
        p_guess = params['p_guess']
        p_1 = params['p_1']
        cost = params['cost']
        cost_exp = params['cost_exp']
        utility_type = params['utility_type']

        st = self.states[s]
        act = self.actions[a]
        st1 = self.states[s1]

        if not st.is_valid_action(act):
            return wlp.NINF
        elif st.term or st1.term:
            return 0
        elif st.is_quiz() and act.name == 'exp':
            return cost_exp
        elif not st.is_quiz() and act.name == 'ask':
            return cost + st.rewards_ask(p_r, p_slip, p_guess, p_1,
                                         utility_type)
        elif not st.is_quiz() and act.is_quiz():
            return cost
        else:
            return 0

    def get_observation(self, s, a, o, params=None, derivative=None):
        """Get observation probability, or derivative

        Args:
            s:          State index (ending)
            a:          Action index
            o:          Observation string
            params:

        """
        if params is None:
            params = self.params
        p_slip = params['p_slip']
        p_guess = params['p_guess']

        act = self.actions[a]
        st = self.states[s]
        obs = self.observations[o]

        def cval(v):
            """Derivative of constant values is always 0"""
            if derivative is None:
                return v
            else:
                return 0

        if st.term:
            # Always know when we enter terminal state.
            if obs == 'term':
                return cval(1)
            else:
                return cval(0)
        elif st.is_quiz() and act.is_quiz() and st.quiz_val == act.quiz_val:
            # Assume teaching actions ask questions that require only the
            # skill being taught.
            p_r_gold = [int(i == st.quiz_val) for i in xrange(self.n_skills)]
            
            has_skills = st.p_has_skills(p_r_gold) == 1
            if has_skills:
                if obs == 'right':
                    if derivative is None:
                        return 1 - p_slip
                    elif derivative == 'p_slip':
                        return -1
                    else:
                        return 0
                elif obs == 'wrong':
                    if derivative is None:
                        return p_slip
                    elif derivative == 'p_slip':
                        return 1
                    else:
                        return 0
                else:
                    return cval(0)
            else:
                if obs == 'right':
                    if derivative is None:
                        return p_guess
                    elif derivative == 'p_guess':
                        return 1
                    else:
                        return 0
                elif obs == 'wrong':
                    if derivative is None:
                        return 1 - p_guess
                    elif derivative == 'p_guess':
                        return -1
                    else:
                        return 0
                else:
                    return cval(0)
        else:
            if obs == 'right':
                return cval(0.5)
            elif obs == 'wrong':
                return cval(0.5)
            else:
                return cval(0)
 

    def make_tables(self, params):
        """Create model tables from parameters
        
        Returns:
            p_t (|S|.|A|.|S| array):        Transition probabilties
            p_o (|S|.|A|.|O| array):        Observation probabilities
            p_i (|S| array):        i       Initial belief
            rewards (|S|.|A|.|S| array):    Rewards
        """
        S = len(self.states)
        A = len(self.actions)
        O = len(self.observations)

        p_t = np.zeros((S, A, S))
        p_o = np.zeros((S, A, O))
        rewards = np.zeros((S, A, S))
        for s in xrange(S):
            for a in xrange(A):
                for s1 in xrange(S):
                    p_t[s][a][s1] = self.get_transition(s, a, s1, params)
                    rewards[s][a][s1] = self.get_reward(s, a, s1, params)

                for o in xrange(O):
                    p_o[s][a][o] = self.get_observation(s, a, o, params)

        # Initial beliefs
        p_i = self.get_start_belief(params)
        
        return p_t, p_o, p_i, rewards 

    def sample_SOR(self, state_num, action_num):
        '''
        Sample a next state, observation, and reward.

        state_num       int
        action_num      int
        return          tuple(s, o, r)
        '''
        p_s_prime = [self.get_transition(state_num, action_num, s_num) for
                     s_num in xrange(len(self.states))]
        s_prime = np.random.choice(range(len(self.states)), p=p_s_prime)
        p_o_prime = [
            self.get_observation(s_prime, action_num, observation_num) for
            observation_num in xrange(len(self.observations))]
        o_prime = np.random.choice(range(len(self.observations)), p=p_o_prime)
        r = self.get_reward(state_num, action_num, s_prime)
        return s_prime, o_prime, r

    def update_belief(self, prev_belief, action_num, observation_num):
        '''
        POMDPModel doesn't store beliefs, so this takes
        and returns a belief vector.

        prev_belief     numpy array
        action_num      int
        observation_num int
        return          numpy array
        '''
        b_new_nonnormalized = []
        for s_prime in xrange(len(self.states)):
            p_o_prime = self.get_observation(
                s_prime, action_num, observation_num)
            summation = 0.0
            for s in xrange(len(self.states)):
                p_s_prime = self.get_transition(s, action_num, s_prime)
                b_s = float(prev_belief[s])
                summation = summation + p_s_prime * b_s
            b_new_nonnormalized.append(p_o_prime * summation)

        # normalize
        b_new = []
        total = sum(b_new_nonnormalized)
        for b_s in b_new_nonnormalized:
            b_new.append(b_s/total)
        return np.array(b_new)

    def expected_sufficient_statistics(self, log_marginals,
                                       log_pairwise_marginals, history):
        """Make tables with expected sufficient statistics

        Args:
            log_marginals:          list of unnormalized log marginals
                                    (np.arrays of (|T+1| x |S|))
            log_pairwise_marginals: list of unnormalized log marginal pairs
                                    (np.arrays of (|T| x |S| x |S|))

        Returns:
            ess_t:  Expected sufficient statistics for transitions.
            ess_o:  Expected sufficient statistics for observations.
            ess_i:  Expected sufficient statistics for initial probabilities.
        """
        S = len(self.states)
        A = len(self.actions)
        O = len(self.observations)
        ess_t = np.zeros((S, A, S))
        ess_o = np.zeros((S, A, O))
        ess_i = np.zeros((S))
        for ep,m in enumerate(log_marginals):
            m_norm = np.exp(m - logsumexp(m, axis=1, keepdims=True))
            T = history.n_t(ep)
            for t in xrange(T):
                for s in xrange(S):
                    a, o = history.get_AO(ep, t)
                    ess_o[s][a][o] += m_norm[t+1][s]
            ess_i += m_norm[0,:]

        for ep,pm in enumerate(log_pairwise_marginals):
            pm_norm = np.exp(pm - logsumexp(pm, axis=(1,2), keepdims=True))
            T = history.n_t(ep)
            for t in xrange(T):
                a, o = history.get_AO(ep, t)
                for s in xrange(S):
                    for s1 in xrange(S):
                        ess_t[s][a][s1] += pm_norm[t][s][s1]
        return ess_t, ess_o, ess_i


    def get_unnormalized_marginals(self, params, history):
        """Estimate unnormalized marginals from provided model parameters
        
        Args:
            params:     
            history:    History object

        Returns:
            tuple(log_marginals, log_pairwise_marginals, log_likelihood):
                log_marginals:          list of unnormalized log marginals
                                        (np.arrays of (|T+1| x |S|))
                log_pairwise_marginals: list of unnormalized log marginal pairs
                                        (np.arrays of (|T| x |S| x |S|))
                log_likelihood:         Log-likelihood 
        
        """
        S = len(self.states)
        ll = 0
        log_marginals = []
        log_pairwise_marginals = []
        for ep in xrange(history.n_episodes()):
            T = history.n_t(ep)
            if T == 0:
                continue

            # Forward-backward init.
            alpha = np.zeros((T+1, S))
            beta = np.zeros((T+1, S))
            for s in xrange(S):
                p_i = self.get_start_probability(s, params)
                alpha[0][s] = log(p_i)
                beta[T][s] = log(1.0)

            # Forward.
            for t in xrange(T):
                a, o = history.get_AO(ep, t)
                for s1 in xrange(S):
                    v = []
                    for s0 in xrange(S):
                        p_t = self.get_transition(s0, a, s1, params)
                        p_o = self.get_observation(s1, a, o, params)
                        v.append(alpha[t][s0] + log(p_t) + log(p_o))
                    alpha[t+1][s1] = logsumexp(v)

            # Backward.
            for t in reversed(xrange(T)):
                a, o = history.get_AO(ep, t)
                for s0 in xrange(S):
                    v = []
                    for s1 in xrange(S):
                        p_t = self.get_transition(s0, a, s1, params)
                        p_o = self.get_observation(s1, a, o, params)
                        v.append(beta[t+1][s1] + log(p_t) + log(p_o))
                    beta[t][s0] = logsumexp(v)

            log_marginals.append(alpha + beta)

            # Make pairwise marginals
            pm = np.zeros((T, S, S))
            for t in xrange(T):
                a, o = history.get_AO(ep, t)
                for s in xrange(S):
                    for s1 in xrange(S):
                        p_t = self.get_transition(s, a, s1, params)
                        p_o = self.get_observation(s1, a, o, params)
                        pm[t][s][s1] = alpha[t][s] + log(p_t) + log(p_o) + \
                                beta[t+1][s1] # BUG: should this be s1 or s
            log_pairwise_marginals.append(pm)

            # Update likelihood
            ll += logsumexp(alpha[T,:])

        return log_marginals, log_pairwise_marginals, ll

    def estimate_E(self, history, params):
        """Get expected sufficient statistics"""
        log_marginals, log_pairwise_marginals, ll = \
            self.get_unnormalized_marginals(params, history)
        ess_t, ess_o, ess_i = self.expected_sufficient_statistics(
            log_marginals, log_pairwise_marginals, history)

        # Add param likelihood.
        arr = self.params_to_array(params)
        for i,beta_p in enumerate(self.params_to_array(self.beta_priors)):
            ll += log(util.beta(arr[i], *beta_p))
 
        return ess_t, ess_o, ess_i, ll

    def get_param_index(self, param):
        return self.n_skills + self.single_params.index(param)

    def params_to_array(self, params):
        return list(params['p_s']) + [params[p] for p in self.single_params]

    def array_to_params(self, array):
        params = dict()
        params['p_s'] = array[:self.n_skills]
        for p in self.single_params:
            params[p] = array[self.get_param_index(p)]
        return params

    def estimate_M(self, ess_t, ess_o, ess_i, last_params=None):
        def f(array):
            params = self.array_to_params(array)
            ell = 0
            grad = np.zeros((self.n_skills + len(self.single_params)))
            S = len(self.states)
            A = len(self.actions)
            O = len(self.observations)
            for s in xrange(S):
                # TODO: Make this a dot product for speedup?
                if ess_i[s] != 0:
                    ell += ess_i[s] * log(self.get_start_probability(s, params))
                    for i in xrange(self.n_skills):
                        grad[i] += \
                            ess_i[s] / self.get_start_probability(s, params) * \
                            self.get_start_probability(s, params, derivative=i)
                for a in xrange(A):
                    for s1 in xrange(S):
                        if ess_t[s][a][s1] != 0:
                            ell += ess_t[s][a][s1] * \
                                   log(self.get_transition(s, a, s1, params))
                            for p in self.single_params:
                                grad[self.get_param_index(p)] += ess_t[s][a][s1] / self.get_transition(s, a, s1, params) * self.get_transition(s, a, s1, params, derivative=p)
                    for o in xrange(O):
                        if ess_o[s][a][o] != 0:
                            ell += ess_o[s][a][o] * \
                                   log(self.get_observation(s, a, o, params))
                            for p in self.single_params:
                                grad[self.get_param_index(p)] += \
                                    ess_o[s][a][o] / \
                                    self.get_observation(s, a, o, params) * \
                                    self.get_observation(s, a, o, params,
                                                         derivative=p)

            for i,beta_p in enumerate(self.params_to_array(self.beta_priors)):
                ell += log(util.beta(array[i], *beta_p))
                grad[i] += 1 / util.beta(array[i], *beta_p) * \
                           util.dbeta(array[i], *beta_p)
        


            return (-1 * ell, -1 * grad)


        if last_params is None:
            x0 = np.random.random((self.n_skills + len(self.single_params)))
        else:
            x0 = self.params_to_array(last_params)
        res = minimize(f, x0, method='L-BFGS-B', jac=True,
                       bounds=[(0.0001,0.9999) for x in x0], options={'disp': False})
        return self.array_to_params(res['x'])

    def estimate(self, history, random_init=False, ll_max_improv=0.001):
        """Estimate parameters from history
        
        Args:
            random_init:    Initialize randomly rather than with last values.
            
        """
        if random_init:
            params = {'p_s': np.random.random((self.n_skills)),
                      'p_guess': random.random(),
                      'p_slip': random.random(),
                      'p_lose': random.random(),
                      'p_learn': random.random(),
                      'p_leave': random.random()}
        else:
            params = dict((k, self.params[k]) for k in 
                          ['p_s'] + self.single_params)

        ess_t, ess_o, ess_i, ll = self.estimate_E(history, params)
        ll_improv = float('inf')
        t = 0
        #print 'EM step {}: {} ({})'.format(t, ll, ll_improv)
        #print params
        while (ll_improv > ll_max_improv):
            t += 1
            params = self.estimate_M(ess_t, ess_o, ess_i, last_params=params) 
            ess_t, ess_o, ess_i, ll_new = self.estimate_E(history, params)
            ll_improv = abs((ll_new - ll) / ll)
            ll = ll_new
            #print 'EM step {}: {} ({})'.format(t, ll, ll_improv)
            #print params

        self.params.update(params)

class POMDPPolicy:
    '''
    Based on mbforbes/py-pomdp on github.

    Read a policy file

    Attributes:
        action_nums    The full list of action (numbers) from the alpha
                       vectors. In other words, this saves the action
                       number from each alpha vector and nothing else,
                       but in the order of the alpha vectors.

        pMatrix        The policy matrix, constructed from all of the
                       alpha vectors.
    '''
    def __init__(self, filename, file_format='policyx', n_states=None):
        if file_format == 'policyx':
            tree = ee.parse(filename)
            root = tree.getroot()
            avec = list(root)[0]
            alphas = list(avec)
            self.action_nums = []
            val_arrs = []
            for alpha in alphas:
                self.action_nums.append(int(alpha.attrib['action']))
                vals = []
                for val in alpha.text.split():
                    vals.append(float(val))
                val_arrs.append(vals)
            if len(val_arrs) == 0:
                raise Exception('APPL policy contained no alpha vectors')
            self.pMatrix = np.array(val_arrs)
        elif file_format == 'aitoolbox':
            # Retrieve max horizon alpha vectors.
            # TODO: Allow retrieval of horizons other than max.
            horizons = [[]]
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('@'):
                        horizons.append([])
                    else:
                        horizons[-1].append(line)
            horizons = [lst for lst in horizons if len(lst) > 0]
            if len(horizons) == 0:
                raise Exception('AIToolbox policy contained no alpha vectors')
            lines_max_horizon = horizons[-1]
            alphas = [[float(v) for v in line.split()[:n_states]] for
                      line in lines_max_horizon]
            self.pMatrix = np.array(alphas)
            self.action_nums = [int(line.split()[n_states]) for
                                line in lines_max_horizon]
        else:
            raise NotImplementedError

    def get_best_action(self, belief):
        '''
        Returns tuple:
            (best-action-num, expected-reward-for-this-action).
        '''
        res = self.pMatrix.dot(belief)
        highest_expected_reward = res.max()
        best_action = self.action_nums[res.argmax()]
        return (best_action, highest_expected_reward)

    def get_action_rewards(self, belief):
        '''
        Returns dictionary:
            action-num: max expected-reward.
        '''
        res = self.pMatrix.dot(belief)
        d = dict()
        for a,r in zip(self.action_nums, res):
            if a not in d:
                d[a] = r
            else:
                d[a] = max(d[a], r)
        return d