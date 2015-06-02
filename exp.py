import multiprocessing as mp
import argparse
import itertools
import csv
import os
import sys
import time
import json
import logging
import numpy as np
import functools as ft
import traceback
from pomdp import POMDPModel
from policy import Policy
from history import History
from util import get_or_default, ensure_dir

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

def run_functor(functor, x):
    """
    Given a no-argument functor, run it and return its result. We can 
    use this with multiprocessing.map and map it over a list of job 
    functors to do them.

    Handles getting more than multiprocessing's pitiful exception output

    https://stackoverflow.com/questions/6126007/
    python-getting-a-traceback-from-a-multiprocessing-process
    """
    try:
        # This is where you do your actual work
        return functor(x)
    except:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def get_start_belief(fp):
    """DEPRECATED"""
    for line in fp:
        if line.strip().startswith('start:'):
            start_belief = np.array([float(s) for s in line[6:].split()])
            return start_belief
    return None

def belief_to_str(lst):
    return ' '.join(str(x) for x in lst)

def set_config_defaults(config):
    """Mutates and returns original object"""
    config['params']['utility_type'] = get_or_default(
        config['params'], 'utility_type', 'acc')
    config['params']['p_lose'] = get_or_default(
        config['params'], 'p_lose', 0)
    config['params']['iterations'] = get_or_default(
        config['params'], 'iterations', 1)
    return config


def run_policy_iteration_from_json(s):
    """Helper for Pool.map(), which can only use functions that take a single
    argument"""
    d = json.loads(s)
    config_params = d['params']
    exp_name = d['exp_name']
    policy = d['policy']
    iteration = d['iteration']
    return run_policy_iteration(exp_name, config_params, policy, iteration)
 
def run_policy_iteration(exp_name, config_params, policy, iteration):
    """
    Args:
        exp_name (str):         Experiment name, without file ending.
        config_params (json):   Params portion of config
        policy (json):
        iteration (int):

    Returns:
        tuple:
            - Experiment rows to be written
            - Model rows to be written
    """
    name_exp = exp_name
    it = iteration

    # Parse config
    params_gt = config_params
    # TODO: Blacklist estimated params instead.
    params_gt_fixed = dict((k, params_gt[k]) for k in
                           ['cost', 'cost_exp', 'p_r', 'p_1', 'utility_type'])
    n_skills = len(params_gt['p_s'])
    episodes = params_gt['episodes']

    pol = Policy(policy_type=policy['type'], exp_name=name_exp, **policy)

    # GT model
    model_gt = POMDPModel(n_skills=n_skills, **params_gt)
    model_actions = model_gt.actions
    model_states = model_gt.states
    model_observations = model_gt.observations

    results = []
    models = []

    # Begin experiment
    history = History()
    if pol.epsilon is not None:
        model_est = POMDPModel(n_skills=n_skills, **params_gt_fixed)
    else:
        model_est = model_gt

    for ep in xrange(episodes):
        logger.info('{} ({}, {})'.format(pol, it, ep))
        # TODO: Move RL logic into Policy class.
        if pol.epsilon is not None and ep % pol.estimate_interval == 0:
            # TODO: Reestimate only for policies that use POMDP solvers?
            first_ep = ep == 0
            model_est.estimate(history, random_init=first_ep)
        params_est = model_est.params
        models += model_est.get_params_est(
            iteration=it, episode=ep, policy=pol)

        history.new_episode()
        start_belief = model_gt.get_start_belief()
        start_state = np.random.choice(range(len(start_belief)),
                                       p=start_belief)
        s = start_state

        # Belief using estimated model.
        belief = model_est.get_start_belief()
        t = 0
        results.append({'iteration': it,
                        'episode': ep,
                        't': t,
                        'policy': str(pol),
                        'sys_t': time.clock(),
                        'a': '',
                        's': s,
                        'o': '',
                        'r': '',
                        'b': belief_to_str(belief)})

        while str(model_states[s]) != 'TERM':
            if (pol.epsilon is not None and
                    np.random.random() <= pol.epsilon):
                valid_actions = [
                    i for i,a in enumerate(model_actions) if
                    model_states[s].is_valid_action(a) and
                    a.name != 'boot']
                a = np.random.choice(valid_actions)
            else:
                a = pol.get_best_action(
                        params_est, it, history, model_states,
                        model_actions, model_observations, belief)

            # Simulate a step
            s, o, r = model_gt.sample_SOR(s, a)
            history.record(a, o)
            belief = model_est.update_belief(belief, a, o)

            t += 1
            results.append({'iteration': it,
                            'episode': ep,
                            't': t,
                            'policy': str(pol),
                            'sys_t': time.clock(),
                            'a': a,
                            's': s,
                            'o': o,
                            'r': r,
                            'b': belief_to_str(belief)})
    return results, models


def run_experiment(config_fp):
    basename = os.path.basename(config_fp.name)
    exp_name = os.path.splitext(basename)[0]

    ensure_dir('res')
    ensure_dir(os.path.join('models', exp_name))
    ensure_dir(os.path.join('policies', exp_name))

    config = json.load(config_fp)
    config = set_config_defaults(config)
    params_gt = config['params']
    n_skills = len(params_gt['p_s'])

    # Prepare worker process arguments
    policies = config['policies']
    iterations = xrange(params_gt['iterations'])
    args_iter = (json.dumps({'exp_name': exp_name,
                             'params': params_gt,
                             'policy': p,
                             'iteration': i}) for i,p in
                 itertools.product(iterations, policies))

    # Write one-time files.
    model_gt = POMDPModel(n_skills=n_skills, **params_gt)
    with open(os.path.join('res', '{}_names.csv'.format(exp_name)), 'wb') as f:
        model_gt.write_names(f)

    models_fp = open(os.path.join('res', '{}_model.csv'.format(exp_name)), 'wb')
    models_fieldnames = ['iteration', 'episode', 'policy', 'param', 'v']
    m_writer = csv.DictWriter(models_fp, fieldnames=models_fieldnames)
    m_writer.writeheader()
    for r in model_gt.get_params_est():
        m_writer.writerow(r)

    results_fp = open(os.path.join('res', '{}.txt'.format(exp_name)), 'wb')
    results_fieldnames = [
        'iteration','episode','t','policy','sys_t','a','s','o','r','b']
    r_writer = csv.DictWriter(results_fp, fieldnames=results_fieldnames)
    r_writer.writeheader()

    # Create worker processes.
    def init_worker():
        """Function to make sure everyone happily exits on KeyboardInterrupt

        See https://stackoverflow.com/questions/1408356/
        keyboard-interrupts-with-pythons-multiprocessing-pool
        """
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(initializer=init_worker)
    f = ft.partial(run_functor, run_policy_iteration_from_json)
    try:
        for res in pool.imap_unordered(f, args_iter):
            results_rows, models_rows = res
            for r in results_rows:
                r_writer.writerow(r)
            for r in models_rows:
                m_writer.writerow(r)
            models_fp.flush()
            results_fp.flush()
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.warn('Control-C pressed')
        pool.terminate()
    finally:
        # Cleanup.
        results_fp.close()
        models_fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('config', type=argparse.FileType('r'), nargs='+', help='Config files')
    args = parser.parse_args()

    for fp in args.config:
        run_experiment(fp)
