"""policy.py

Requirements: $PATH must include pomdpsol-appl for 'appl' policies and
pomdpsol-aitoolbox for 'aitoolbox' policies.

"""

from __future__ import division
import collections
import os
import time
import copy
import random
import numpy as np
import subprocess
from pomdp import POMDPPolicy, POMDPModel
import util
from util import get_or_default, ensure_dir, equation_safe_filename
import work_learn_problem as wlp

class Policy:
    """Policy class

    Assumes policy files for appl policies live in relative folder 'policies'

    """
    def __init__(self, policy_type, exp_name, n_worker_classes, params_gt,
                 **kwargs):
        default_discount = 0.99
        self.policy = policy_type
        self.exp_name = exp_name
        self.epsilon = get_or_default(kwargs, 'epsilon', None)
        self.explore_actions = get_or_default(kwargs, 'explore_actions', None)
        self.thompson = bool(get_or_default(kwargs, 'thompson', False))
        self.hyperparams = get_or_default(kwargs, 'hyperparams', None)
        if self.rl_p():
            self.resolve_interval = get_or_default(
                kwargs, 'resolve_interval', 1)
            self.estimate_interval = get_or_default(
                kwargs, 'estimate_interval', self.resolve_interval)
        if self.policy in ('appl', 'zmdp'):
            self.discount = get_or_default(kwargs, 'discount', default_discount)
            self.timeout = get_or_default(kwargs, 'timeout', None)
        elif self.policy == 'aitoolbox':
            self.discount = get_or_default(kwargs, 'discount', default_discount)
            self.horizon = kwargs['horizon']
        elif self.policy == 'teach_first':
            self.n = kwargs['n']
            self.teach_type = kwargs['teach_type']
        elif self.policy == 'test_and_boot':
            self.n_test = kwargs['n_test']
            self.n_work = kwargs['n_work']
            self.accuracy = kwargs['accuracy']
        else:
            raise NotImplementedError

        if self.rl_p():
            name = kwargs['hyperparams']
            mod = __import__('param', fromlist=[name])
            cls = getattr(mod, name)
            self.model = POMDPModel(
                n_worker_classes, params=params_gt,
                hyperparams=cls(params_gt, n_worker_classes),
                estimate_all=True)
        else:
            self.model = POMDPModel(n_worker_classes, params=params_gt)

        self.params_estimated = dict()
        self.hparams_estimated = dict()
        self.estimate_times = dict()
        self.resolve_times = dict()
        self.external_policy = None

    def rl_p(self):
        """Policy does reinforcement learning"""
        return self.epsilon is not None or self.thompson
    
    def get_epsilon_probability(self, worker, t):
        """Return probability specified by the given exploration function.

        Exploration function is a function of the worker (w or worker)
        and the current timestep (t).

        WARNING: Evaluates the expression in self.epsilon without security checks.

        """
        # Put some useful variable abbreviations in the namespace.
        w = worker
        if isinstance(self.epsilon, basestring):
            return eval(self.epsilon)
        else:
            return self.epsilon

    def estimate_and_solve(self, iteration, history):
        """Reestimate and resolve as needed."""
        worker = history.n_workers() - 1

        resolve_p = (self.policy in ('appl', 'zmdp', 'aitoolbox') and
                     (self.external_policy is None or
                      (self.rl_p() and
                       worker % self.resolve_interval == 0)))
        estimate_p = (self.rl_p() and
                      (resolve_p or worker % self.estimate_interval == 0))
        model = self.model
        if estimate_p:
            start = time.clock()
            model.estimate(history,
                           random_init=(len(self.params_estimated) == 0))
            if self.thompson:
                model.thompson_sample()
            self.estimate_times[worker] = time.clock() - start
            self.params_estimated[worker] = copy.deepcopy(
                model.get_params_est())
            self.hparams_estimated[worker] = copy.deepcopy(model.hparams)
        if resolve_p:
            utime1, stime1, cutime1, cstime1, _ = os.times()
            self.external_policy = self.get_external_policy(iteration, worker)
            utime2, stime2, cutime2, cstime2, _ = os.times()
            # All solvers are run as subprocesses, so count elapsed
            # child process time.
            self.resolve_times[worker] = cutime2 - cutime1 + \
                                         cstime2 - cstime1

    def get_next_action(self, iteration, history, valid_actions, belief=None):
        """Get next action, according to policy.

        If this is an RL policy, take random action (other than boot)
        with probability specified by the given exploration function.

        """
        worker = history.n_workers() - 1
        t = history.n_t(worker)
        if (self.epsilon is not None and
                np.random.random() <= self.get_epsilon_probability(worker, t)):
            valid_explore_actions = [
                i for i in valid_actions if
                self.model.actions[i].get_type() in self.explore_actions]
            return np.random.choice(valid_explore_actions)
        else:
            return self.get_best_action(iteration, history,
                                        valid_actions, belief)


    def get_best_action(self, iteration, history, valid_actions, belief=None):
        """Get best action according to policy.

        If policy requires an external_policy, assumes it already exists.

        Args:
            iteration (int):            Current iteration.
            valid_actions (lst):        Indices of valid actions.
            history (History object):   Defined in history.py.

        Returns: Action index.

        """
        model = self.model
        a_ask = model.actions.index(wlp.Action('ask'))
        a_boot = model.actions.index(wlp.Action('boot'))
        worker = history.n_workers() - 1
        current_AO = history.history[-1]
        if len(current_AO) == 0:
            current_actions = []
            current_observations = []
        else:
            current_actions, current_observations = zip(*current_AO)
        n_skills = model.n_skills
        n_actions = len(current_actions)
        if self.policy == 'teach_first':
            # Make sure to teach each skill at least n times.
            # Select skills in random order, but teach each skill as a batch.
            if self.teach_type == 'exp':
                teach_actions = [i for i, a in enumerate(model.actions) if
                                 a.is_quiz()]
                teach_counts = collections.defaultdict(int)
                for i in xrange(len(current_actions) - 1):
                    a1 = model.actions[current_actions[i]]
                    a2 = model.actions[current_actions[i + 1]]
                    if a1.is_quiz() and a2.name == 'exp':
                        teach_counts[current_actions[i]] += 1
            elif self.teach_type == 'tell':
                teach_actions = [i for i, a in enumerate(model.actions) if
                                 a.name == 'tell']
                teach_counts = collections.Counter(
                    [a for a in current_actions if a in teach_actions])
            teach_actions_remaining = [a for a in teach_actions if
                                       teach_counts[a] < self.n]
            teach_actions_in_progress = [a for a in teach_actions_remaining if
                                         teach_counts[a] > 0]
            if n_actions == 0:
                if teach_actions_remaining:
                    return random.choice(teach_actions_remaining)
                else:
                    return a_ask
            else:
                last_action = current_actions[-1]
                if (self.teach_type == 'exp' and
                        last_action in teach_actions_remaining):
                    return model.actions.index(wlp.Action('exp'))
                else:
                    if len(teach_actions_in_progress) > 0:
                        return random.choice(teach_actions_in_progress)
                    elif len(teach_actions_remaining) > 0:
                        return random.choice(teach_actions_remaining)
                    else:
                        return a_ask
        elif self.policy == 'test_and_boot':
            n_work_actions = len([a for a in current_actions if
                                  a == a_ask])
            last_action_block = util.last_true(
                current_actions, lambda a: model.actions[a].is_quiz())
            test_counts = collections.Counter(last_action_block)
            if n_work_actions % self.n_work == 0:
                test_actions = [i for i, a in enumerate(model.actions) if
                                a.is_quiz()]
                test_actions_remaining = [a for a in test_actions if
                                          test_counts[a] < self.n_test]
                if len(test_actions_remaining) == 0:
                    # Testing done. Check accuracy.
                    test_answers = current_observations[
                        -1 * len(last_action_block):]
                    assert all(model.observations[i] in ['wrong', 'right'] for
                               i in test_answers)
                    accuracy = sum([model.observations[i] == 'right' for
                                    i in test_answers]) / len(test_answers)
                    if accuracy >= self.accuracy:
                        return a_ask
                    else:
                        return a_boot
                else:
                    return random.choice(test_actions_remaining)
            else:
                return a_ask
        elif self.policy in ('appl', 'aitoolbox', 'zmdp'):
            rewards = self.external_policy.get_action_rewards(belief)
            valid_actions_with_rewards = set(valid_actions).intersection(
                set(rewards))
            if len(valid_actions_with_rewards) == 0:
                raise Exception('No valid actions in policy')
            max_reward = max(rewards.itervalues())
            valid_rewards = dict((a, rewards[a]) for a in valid_actions_with_rewards)
            max_valid_reward = max(valid_rewards.itervalues())
            if max_reward > max_valid_reward:
                print 'Warning: best reward not available'
            # Take random best action.
            best_valid_action = random.choice(
                [a for a in valid_rewards if
                 valid_rewards[a] == max_valid_reward])
            return best_valid_action
        else:
            raise NotImplementedError

    def get_external_policy(self, iteration, worker):
        """Compute external policy and store in unique locations.
        
        Store POMDP files as
        'models/exp_name/iteration/policy_name-worker.pomdp'.

        Store learned policy files as
        'policies/exp_name/iteration/policy_name-worker.policy'.

        Returns:
            policy (POMDPPolicy)

        """
        pomdp_dirpath = os.path.join(
            'models', self.exp_name, str(iteration))
        policy_dirpath = os.path.join(
            'policies', self.exp_name, str(iteration))
        ensure_dir(pomdp_dirpath)
        ensure_dir(policy_dirpath)
        pomdp_fpath = os.path.join(
            pomdp_dirpath, '{}-{:06d}.pomdp'.format(self, worker))
        policy_fpath = os.path.join(
            policy_dirpath, '{}-{:06d}.policy'.format(self, worker))

        return self.run_solver(model_filename=pomdp_fpath,
                               policy_filename=policy_fpath)

    def run_solver(self, model_filename, policy_filename):
        """Run POMDP solver.
        
        Args:
            model_filename (str):       Path for input to POMDP solver.
            policy_filename (str):      Path for computed policy.

        Returns:
            policy (POMDPPolicy)

        """
        model = self.model
        if self.policy == 'appl':
            with open(model_filename, 'w') as f:
                model.write_pomdp(f, discount=self.discount)
            args = ['pomdpsol-appl',
                    model_filename,
                    '-o', policy_filename]
            if self.timeout is not None:
                args += ['--timeout', str(self.timeout)]
            subprocess.call(args)
            return POMDPPolicy(policy_filename,
                               file_format='policyx')
        elif self.policy == 'aitoolbox':
            with open(model_filename, 'w') as f:
                model.write_txt(f)
            args = ['pomdpsol-aitoolbox',
                    '--input', model_filename,
                    '--output', policy_filename,
                    '--discount', str(self.discount),
                    '--horizon', str(self.horizon),
                    '--n_states', str(len(model.states)),
                    '--n_actions', str(len(model.actions)),
                    '--n_observations', str(len(model.observations))]
            subprocess.call(args)
            return POMDPPolicy(policy_filename,
                               file_format='aitoolbox',
                               n_states=len(model.states))
        elif self.policy == 'zmdp':
            with open(model_filename, 'w') as f:
                model.write_pomdp(f, discount=self.discount)
            args = ['pomdpsol-zmdp',
                    'solve', model_filename,
                    '-o', policy_filename]
            if self.timeout is not None:
                args += ['-t', str(self.timeout)]
            subprocess.call(args)
            return POMDPPolicy(policy_filename,
                               file_format='zmdp',
                               n_states=len(model.states))


    def get_valid_actions(self, belief, states, actions):
        """Return valid actions given the current belief.


        Return actions valid from the first state with non-zero probability.
        """
        for p,s in zip(belief, states):
            if p > 0:
                return [i for i,a in enumerate(actions) if s.is_valid_action(a)]
        raise Exception('Invalid belief state')
             

    def __str__(self):
        if self.policy in ('appl', 'zmdp'):
            s = self.policy + '-d{:.3f}'.format(self.discount)
            if self.timeout is not None:
                s += '-tl{}'.format(self.timeout)
        elif self.policy == 'aitoolbox':
            s = 'ait' + '-d{:.3f}'.format(self.discount)
            s += '-h{}'.format(self.horizon)
        elif self.policy == 'teach_first':
            s = self.policy + '-n_{}_{}'.format(self.teach_type, self.n)
        elif self.policy == 'test_and_boot':
            s = self.policy + '-n_test_{}-n_work_{}-acc_{}'.format(
                    self.n_test, self.n_work, self.accuracy)
        else:
            raise NotImplementedError

        if self.rl_p():
            if self.epsilon is not None:
                s += '-eps_{}'.format(equation_safe_filename(self.epsilon))
                s += '-explore_{}'.format('_'.join(self.explore_actions))
            if self.thompson:
                s += '-thomp'
            if self.hyperparams and self.hyperparams != 'HyperParams':
                s += '-{}'.format(self.hyperparams)
            if (self.estimate_interval > 1):
                s += '-e_int{}'.format(self.estimate_interval)
            if (self.resolve_interval > 1):
                s += '-s_int{}'.format(self.resolve_interval)
        return s
