import multiprocessing as mp
import argparse
import csv
import os
import subprocess
import collections # for Policy
import random # for Policy
import time
import json
import numpy as np
import work_learn_problem as wlp
import pypomdp as pp
from make_pomdp import POMDPWriter
from util import get_or_default, ensure_dir

APPL_PATH = '/homes/gws/jbragg/dev/appl-0.96/'

def get_start_belief(fp):
    """DEPRECATED"""
    for line in fp:
        if line.strip().startswith('start:'):
            start_belief = np.array([float(s) for s in line[6:].split()])
            return start_belief
    return None

def belief_to_str(lst):
    return ' '.join(str(x) for x in lst)

def write_names(fp, model):
    """Write csv file with mapping from indices to names."""
    writer = csv.writer(fp)
    writer.writerow(['i','type','s'])
    for i,a in enumerate(model.actions):
        writer.writerow([i, 'action', a])
    for i,s in enumerate(model.states):
        writer.writerow([i, 'state', s])
    for i,o in enumerate(model.observations):
        writer.writerow([i, 'observation', o])


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
            params (dict):          Current params.
            history (dict):         Contains 'actions' and 'observations',
                                    each of which is a list of lists
                                    containing values.
            states (list):          State objects.
            actions (list):         Action objects.
            observations (list):    Observation names.

        Returns: Action index.

        """
        # TODO: Move episode computation to history class.
        episode = len(history['actions']) - 1
        current_actions = history['actions'][-1]
        current_observations = history['observations'][-1]
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
                self.get_external_policy(iteration, episode, params,
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

    def get_external_policy(self, iteration, episode, params,
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
            writer = POMDPWriter(**params)
            if self.policy == 'appl':
                with open(pomdp_fpath, 'w') as f:
                    writer.write_pomdp(f, discount=self.discount)
                args = [os.path.join(APPL_PATH, 'src', 'pomdpsol'),
                        pomdp_fpath,
                        '-o', policy_fpath]
                if self.timeout is not None:
                    args += ['--timeout', str(self.timeout)]
                subprocess.call(args)
                self.external_policy = pp.POMDPPolicy(policy_fpath,
                                                      file_format='policyx')
            elif self.policy == 'aitoolbox':
                with open(pomdp_fpath, 'w') as f:
                    writer.write_txt(f)
                args = [os.path.join('bin', 'pomdpsol'),
                        '--input', pomdp_fpath,
                        '--output', policy_fpath,
                        '--discount', str(self.discount),
                        '--horizon', str(self.horizon),
                        '--n_states', str(len(states)),
                        '--n_actions', str(len(actions)),
                        '--n_observations', str(len(observations))]
                subprocess.call(args)
                self.external_policy = pp.POMDPPolicy(policy_fpath,
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
            s += '-e{.3f}'.format(self.epsilon)
        return s



def run_experiment(config_fp):
    basename = os.path.basename(config_fp.name)
    name_exp = os.path.splitext(basename)[0]

    # Parse config
    config = json.load(config_fp)
    params_gt = config['params']
    iterations = get_or_default(params_gt, 'iterations', 1)
    episodes = params_gt['episodes']
    policies = [Policy(policy_type=p['type'],
                       exp_name=name_exp, **p) for p in config['policies']]


    # Always write a 0.99 discount ground truth .pomdp file.
    ensure_dir(os.path.join('models', name_exp, 'gt'))
    model_writer_gt = POMDPWriter(**params_gt)
    model_gt_path = os.path.join('models', name_exp, 'gt', 'gt.pomdp')
    with open(model_gt_path, 'w') as f:
        model_writer_gt.write_pomdp(f, discount=0.99)
    start_belief = model_writer_gt.get_start_belief()
    model_actions = model_writer_gt.actions
    model_states = model_writer_gt.states
    model_observations = model_writer_gt.observations
    model_gt = pp.POMDPEnvironment(model_gt_path)

    #ensure_dir(os.path.join('policies', name_exp))
    # TODO: Why doesn't this work without the name_exp part?
    ensure_dir(os.path.join('res', name_exp))

    #filename_policy = '../appl-0.96/src/{}.policy'.format(name_exp)
    #filename_env = '{}.pomdp'.format(name_exp)
    #pomdp = pp.POMDP(filename_env, filename_policy, start_belief)

    with open(os.path.join('res', '{}_names.csv'.format(name_exp)), 'wb') as f:
        write_names(f, model_gt)

    results_fields = ['iteration','episode','t','policy','sys_t',
                      'a','s','o','r','b']
    results_fp = open(os.path.join('res', '{}.txt'.format(name_exp)), 'wb')
    r_writer = csv.DictWriter(results_fp, fieldnames=results_fields)
    r_writer.writeheader()

    for it in xrange(iterations):
        for pol in policies:
            # TODO: Clear policy for new run.
            history = {'actions': [], 'observations': []}
            for ep in xrange(episodes):
                # Initialize history
                # TODO: Make this a class.
                history['actions'].append([])
                history['observations'].append([])

                start_state = np.random.choice(range(len(start_belief)),
                                               p=start_belief)
                # Belief using ground truth model.
                belief = start_belief
                s = start_state
                t = 0
                r_writer.writerow({'iteration': it,
                                   'episode': ep,
                                   't': t,
                                   'policy': pol,
                                   'sys_t': time.clock(),
                                   'a': '',
                                   's': s,
                                   'o': '',
                                   'r': '',
                                   'b': belief_to_str(belief)})

                while str(model_states[s]) != 'TERM':
                    # TODO: Give belief computed with current model params.
                    a = pol.get_best_action(params_gt, it, history, model_states, model_actions, model_observations, belief)

                    # Simulate a step
                    s, o, r = model_gt.sample_SOR(s, a)
                    history['actions'][-1].append(a)
                    history['observations'][-1].append(o)
                    belief = model_gt.update_belief(belief, a, o)

                    t += 1
                    r_writer.writerow({'iteration': it,
                                       'episode': ep,
                                       't': t,
                                       'policy': pol,
                                       'sys_t': time.clock(),
                                       'a': a,
                                       's': s,
                                       'o': o,
                                       'r': r,
                                       'b': belief_to_str(belief)})

    results_fp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('config', type=argparse.FileType('r'), nargs='+', help='Config files')
    args = parser.parse_args()

    jobs = []
    for fp in args.config:
        p = mp.Process(target=run_experiment, args=(fp,))
        jobs.append(p)
        p.start()
