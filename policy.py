"""policy.py

Requirements: $PATH must include pomdpsol-appl for 'appl' policies and
pomdpsol-aitoolbox for 'aitoolbox' policies.

"""

import collections
import os
import random
import subprocess
from pomdp import POMDPPolicy, POMDPModel
from util import get_or_default, ensure_dir
import work_learn_problem as wlp

class Policy:
    """Policy class

    Assumes policy files for appl policies live in relative folder 'policies'

    """
    def __init__(self, policy_type, exp_name, **kwargs):
        default_discount = 0.99
        self.policy = policy_type
        self.exp_name = exp_name
        self.epsilon = get_or_default(kwargs, 'epsilon', None)
        self.external_policy_set = collections.defaultdict(bool)
        if self.epsilon is not None:
            self.estimate_interval = get_or_default(
                kwargs, 'estimate_interval', 1)
            self.resolve_interval = get_or_default(
                kwargs, 'resolve_interval', self.estimate_interval)
        if self.policy in ('appl', 'zmdp'):
            self.discount = get_or_default(kwargs, 'discount', default_discount)
            self.timeout = get_or_default(kwargs, 'timeout', None)
        elif self.policy == 'aitoolbox':
            self.discount = get_or_default(kwargs, 'discount', default_discount)
            self.horizon = kwargs['horizon']
        elif self.policy == 'fixed':
            self.n = kwargs['n']
        else:
            raise NotImplementedError

    def get_best_action(self, params, iteration, history, states, actions, observations, valid_actions, belief=None):
        """Next action

        Args:
            params (dict):              Current params.
            history (History object):   Defined in history.py.
            states (list):              State objects.
            actions (list):             Action objects.
            observations (list):        Observation names.

        Returns: Action index.

        """
        # TODO: Move episode computation to history class.
        episode = history.n_episodes() - 1
        current_actions = history.get_actions(episode)
        current_observations = history.get_observations(episode)
        n_skills = len(params['p_s'])
        n_actions = len(current_actions)
        if self.policy == 'fixed':
            # Make sure to teach each skill at least n times.
            # Select skills in random order, but teach each skill as a batch.
            # Alternate quizzing and explaining.
            a_exp = actions.index(wlp.Action('exp'))
            a_ask = actions.index(wlp.Action('ask'))
            quiz_actions = [i for i,a in enumerate(actions) if a.is_quiz()]
            quiz_counts = collections.Counter(
                [a for a in current_actions if a in quiz_actions])
            quiz_actions_remaining = [a for a in quiz_actions if
                                      quiz_counts[a] < self.n]
            quiz_actions_in_progress = [a for a in quiz_actions_remaining if
                                        quiz_counts[a] > 0]
            if n_actions == 0:
                if quiz_actions_remaining:
                    return random.choice(quiz_actions_remaining)
                else:
                    return a_ask
            else:
                last_action = current_actions[-1]
                if actions[last_action].is_quiz():
                    if quiz_counts[last_action] <= self.n:
                        # Explain n times for each quiz.
                        return a_exp
                    else:
                        # We happened to get to a quiz state, so take a
                        # random action.
                        #
                        # What we really want to do here is take EXP or NOEXP
                        # and this is equivalent since booting isn't allowed
                        # from quiz states.
                        return random.choice(valid_actions)
                else:
                    if len(quiz_actions_in_progress) > 0:
                        return random.choice(quiz_actions_in_progress)
                    elif len(quiz_actions_remaining) > 0:
                        return random.choice(quiz_actions_remaining)
                    else:
                        return a_ask
        elif self.policy in ('appl', 'aitoolbox', 'zmdp'):
            resolve_p = (not self.external_policy_set[episode] or
                         (self.epsilon is not None and
                          episode % self.resolve_interval == 0))
            if resolve_p:
                self.external_policy = self.get_external_policy(
                    iteration, episode, params)
                self.external_policy_set[episode] = True
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

    def get_external_policy(self, iteration, episode, params):
        """Compute external policy and store in unique locations.
        
        Store POMDP files as
        'models/exp_name/iteration-episode-policy_name.pomdp'.

        Store learned policy files as
        'policies/exp_name/iteration-episode-policy_name.policy'.

        Returns:
            policy (POMDPPolicy)

        """
        pomdp_dirpath = os.path.join('models', self.exp_name)
        policy_dirpath = os.path.join('policies', self.exp_name)
        ensure_dir(pomdp_dirpath)
        ensure_dir(policy_dirpath)
        pomdp_fpath = os.path.join(
            pomdp_dirpath,
            '{}-{}-{}.pomdp'.format(iteration, episode, self))
        policy_fpath = os.path.join(
            policy_dirpath,
            '{}-{}-{}.policy'.format(iteration, episode, self))

        return self.run_solver(model_filename=pomdp_fpath,
                               policy_filename=policy_fpath,
                               params=params)

    def run_solver(self, model_filename, policy_filename, params):
        """Run POMDP solver, storing files at the given locations

        Returns:
            policy (POMDPPolicy)

        """
        model = POMDPModel(**params)
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
        elif self.policy == 'fixed':
            s = self.policy + '-n{}'.format(self.n)
        else:
            raise NotImplementedError

        if self.epsilon is not None:
            s += '-e{:.3f}'.format(self.epsilon)
            if (self.estimate_interval > 1):
                s += '-e_int{}'.format(self.estimate_interval)
            if (self.resolve_interval > 1):
                s += '-s_int{}'.format(self.resolve_interval)
        return s
