import collections
import os
import random
import subprocess
from pomdp import POMDPPolicy, POMDPModel
from util import get_or_default, ensure_dir
import work_learn_problem as wlp

APPL_PATH = '/homes/gws/jbragg/dev/appl-0.96/'

class Policy:
    """Policy class

    Assumes policy files for appl policies live in relative folder 'policies'

    """
    def __init__(self, policy_type, exp_name, **kwargs):
        default_discount = 0.99
        self.policy = policy_type
        self.exp_name = exp_name
        self.epsilon = get_or_default(kwargs, 'epsilon', None)
        if self.policy == 'appl':
            self.discount = get_or_default(kwargs, 'discount', default_discount)
            self.timeout = get_or_default(kwargs, 'timeout', None)
        elif self.policy == 'aitoolbox':
            self.discount = get_or_default(kwargs, 'discount', default_discount)
            self.horizon = kwargs['horizon']
        elif self.policy == 'fixed':
            self.n = kwargs['n']
        else:
            raise NotImplementedError

    def get_best_action(self, params, iteration, history, states, actions, observations, belief=None):
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
            # Teach each skill n times.
            # Select skills in random order, but teach each skill as a batch.
            # Alternate quizzing and explaining.
            if n_actions < n_skills * self.n * 2:
                a_exp = actions.index(wlp.Action('exp'))
                if n_actions % 2 == 1:
                    return a_exp
                action_counts = collections.Counter(current_actions)
                in_progress = [k for k in action_counts if
                               k != a_exp and action_counts[k] < self.n]
                if in_progress:
                    assert len(in_progress) == 1
                    return in_progress[0]
                else:
                    skill_actions = [i for i,a in enumerate(actions) if
                                     a.is_quiz()]
                    next_action = random.choice([i for i in skill_actions if
                                                i not in action_counts])
                    return next_action
            else:
                return actions.index(wlp.Action('ask'))
        elif self.policy in ('appl', 'aitoolbox'):
            if n_actions == 0:
                self.get_external_policy(iteration, episode, params, n_skills,
                                         states=states, actions=actions,
                                         observations=observations)
            rewards = self.external_policy.get_action_rewards(belief)
            valid_actions = self.get_valid_actions(belief, states, actions)
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

    def get_external_policy(self, iteration, episode, params, n_skills,
                            states, actions, observations):
        """Load external policy (recompute if necessary)
        
        Store POMDP files as
        'models/exp_name/iteration/episode/policy_name.pomdp',
        or as 'models/exp_name/gt/policy_name.pomdp when not using RL.

        Store learned policy files in same way, except prefixed with
        'policies/exp_name/iteration/episode/policy_name.policy',
        or as 'policies/exp_name/gt/policy_name.policy when not using RL.

        """
        if self.epsilon is None:
            pomdp_dirpath = os.path.join('models', self.exp_name, 'gt')
            policy_dirpath = os.path.join('policies', self.exp_name, 'gt')
        else:
            pomdp_dirpath = os.path.join(
                'models', self.exp_name, str(iteration), str(episode))
            policy_dirpath = os.path.join(
                'policies', self.exp_name, str(iteration), str(episode))
        ensure_dir(pomdp_dirpath)
        ensure_dir(policy_dirpath)
        pomdp_fpath = os.path.join(pomdp_dirpath, str(self) + '.pomdp')
        policy_fpath = os.path.join(policy_dirpath, str(self) + '.policy')

        if not (self.epsilon is None and iteration + episode > 0):
            # Recompute always, except only once for non-RL policies.
            model = POMDPModel(n_skills=n_skills, **params)
            if self.policy == 'appl':
                with open(pomdp_fpath, 'w') as f:
                    model.write_pomdp(f, discount=self.discount)
                args = [os.path.join(APPL_PATH, 'src', 'pomdpsol'),
                        pomdp_fpath,
                        '-o', policy_fpath]
                if self.timeout is not None:
                    args += ['--timeout', str(self.timeout)]
                subprocess.call(args)
                self.external_policy = POMDPPolicy(policy_fpath,
                                                   file_format='policyx')
            elif self.policy == 'aitoolbox':
                with open(pomdp_fpath, 'w') as f:
                    model.write_txt(f)
                args = [os.path.join('bin', 'pomdpsol'),
                        '--input', pomdp_fpath,
                        '--output', policy_fpath,
                        '--discount', str(self.discount),
                        '--horizon', str(self.horizon),
                        '--n_states', str(len(states)),
                        '--n_actions', str(len(actions)),
                        '--n_observations', str(len(observations))]
                subprocess.call(args)
                self.external_policy = POMDPPolicy(policy_fpath,
                                                   file_format='aitoolbox',
                                                   n_states=len(states))
            else:
                raise NotImplementedError


    def get_valid_actions(self, belief, states, actions):
        """Return valid actions given the current belief.


        Return actions valid from the first state with non-zero probability.
        """
        for p,s in zip(belief, states):
            if p > 0:
                return [i for i,a in enumerate(actions) if s.is_valid_action(a)]
        raise Exception('Invalid belief state')
             

    def __str__(self):
        if self.policy == 'appl':
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
            s += '-e{:3f}'.format(self.epsilon)
        return s
