from __future__ import division
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
from util import get_or_default, ensure_dir, equation_safe_filename

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
    config['utility_type'] = get_or_default(config, 'utility_type', 'acc')
    config['p_lose'] = get_or_default(config, 'p_lose', 0)
    config['episodes'] = get_or_default(config, 'episodes', 1)
    return config

def params_to_rows(params, iteration=None, episode=None, policy=None):
    """Convert params to list of dictionaries to write to models file"""
    rows = []
    row_base = {'iteration': iteration,
                'episode': episode,
                'policy': policy}
    for p in params:
        if p == 'p_s':
            for i in xrange(len(params[p])):
                row = {'param': 'p_s{}'.format(i), 'v': params['p_s'][i]}
                row.update(row_base)
                rows.append(row)
        else:
            row = {'param': p, 'v': params[p]}
            row.update(row_base)
            rows.append(row)
    return rows

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
    it = iteration

    # Parse config
    params_gt = config_params
    # TODO: Blacklist estimated params instead.
    params_gt_fixed = dict((k, params_gt[k]) for k in
                           ['cost', 'cost_exp', 'p_r', 'p_1', 'utility_type'])
    episodes = params_gt['episodes']

    pol = Policy(policy_type=policy['type'], exp_name=exp_name, **policy)

    # GT model
    model_gt = POMDPModel(**params_gt)

    results = []

    # Begin experiment
    history = History()
    if pol.epsilon is not None:
        model_est = POMDPModel(**params_gt_fixed)
    else:
        model_est = model_gt

    for ep in xrange(episodes):
        logger.info('{} ({}, {})'.format(pol, it, ep))
        history.new_episode()
        start_belief = model_gt.get_start_belief()
        start_state = np.random.choice(range(len(start_belief)),
                                       p=start_belief)
        s = start_state

        # Belief using estimated model.
        pol.estimate_and_solve(model_est, iteration, history)
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

        while str(model_gt.states[s]) != 'TERM':
            valid_actions = [i for i,a in enumerate(model_gt.actions) if
                             model_gt.states[s].is_valid_action(a)]
            a = pol.get_next_action(model_est, it, history, valid_actions, belief)

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

    # Record models, estimate times, and resolve times.
    models = []
    for ep in sorted(pol.params_estimated):
        params = pol.params_estimated[ep]
        models += params_to_rows(params, iteration, ep, str(pol))

    timings = []
    for ep in sorted(pol.estimate_times):
        timings.append({'iteration': iteration,
                        'episode': ep,
                        'policy': str(pol),
                        'type': 'estimate',
                        'duration': pol.estimate_times[ep]})

    for ep in sorted(pol.resolve_times):
        timings.append({'iteration': iteration,
                        'episode': ep,
                        'policy': str(pol),
                        'type': 'resolve',
                        'duration': pol.resolve_times[ep]})

    return results, models, timings


def run_experiment(config, policies, iterations, epsilon, resolve_interval):
    config_basename = os.path.basename(config.name)
    exp_name = os.path.splitext(config_basename)[0]
    policies_basename = os.path.basename(policies.name)
    # TODO: Add epsilon and resolve_interval to policies name.
    policies_name = os.path.splitext(policies_basename)[0]
    if epsilon is not None:
        policies_name += '-eps_{}'.format(equation_safe_filename(epsilon))
    if resolve_interval is not None:
        policies_name += '-s_int{}'.format(resolve_interval)

    res_path = os.path.join('res', exp_name)
    models_path = os.path.join('models', exp_name)
    policies_path = os.path.join('policies', exp_name)
    for d in [res_path, models_path, policies_path]:
        ensure_dir(d)

    params_gt = json.load(config)
    params_gt = set_config_defaults(params_gt)

    # Make folders (errors when too many folders are made in subprocesses).
    for i in xrange(iterations):
        for ep in xrange(params_gt['episodes']):
            ensure_dir(os.path.join(models_path, str(i), str(ep)))
            ensure_dir(os.path.join(policies_path, str(i), str(ep)))

    # Prepare worker process arguments
    policies = json.load(policies)
    for p in policies:
        if epsilon is not None and 'epsilon' not in p:
            p['epsilon'] = epsilon
        if resolve_interval is not None and 'resolve_interval' not in p:
            p['resolve_interval'] = resolve_interval
    args_iter = (json.dumps({'exp_name': exp_name,
                             'params': params_gt,
                             'policy': p,
                             'iteration': i}) for i,p in
                 itertools.product(xrange(iterations), policies))

    # Write one-time files.
    model_gt = POMDPModel(**params_gt)
    with open(os.path.join(res_path, '{}_names.csv'.format(policies_name)), 'wb') as f:
        model_gt.write_names(f)

    # Open file pointers.
    models_fp = open(
        os.path.join(res_path, '{}_model.csv'.format(policies_name)), 'wb')
    models_fieldnames = ['iteration', 'episode', 'policy', 'param', 'v']
    m_writer = csv.DictWriter(models_fp, fieldnames=models_fieldnames)
    m_writer.writeheader()

    for r in  params_to_rows(model_gt.get_params_est()):
        m_writer.writerow(r)

    results_fp = open(os.path.join(res_path, '{}.txt'.format(policies_name)), 'wb')
    results_fieldnames = [
        'iteration','episode','t','policy','sys_t','a','s','o','r','b']
    r_writer = csv.DictWriter(results_fp, fieldnames=results_fieldnames)
    r_writer.writeheader()

    timings_fp = open(
        os.path.join(res_path, '{}_timings.csv'.format(policies_name)), 'wb')
    timings_fieldnames = ['iteration', 'episode', 'policy', 'type', 'duration']
    t_writer = csv.DictWriter(timings_fp, fieldnames=timings_fieldnames)
    t_writer.writeheader()

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
            results_rows, models_rows, timings_rows = res
            for r in results_rows:
                r_writer.writerow(r)
            for r in models_rows:
                m_writer.writerow(r)
            for r in timings_rows:
                t_writer.writerow(r)
            models_fp.flush()
            results_fp.flush()
            timings_fp.flush()
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.warn('Control-C pressed')
        pool.terminate()
    finally:
        # Cleanup.
        results_fp.close()
        models_fp.close()
        timings_fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('--config', '-c', type=argparse.FileType('r'), required=True, help='Config json file')
    parser.add_argument('--policies', '-p', type=argparse.FileType('r'), required=True, help='Policies json file')
    parser.add_argument('--iterations', '-i', type=int, default=400, help='Number of iterations')
    parser.add_argument('--epsilon', type=str, help='Epsilon to use for all policies')
    parser.add_argument('--resolve_interval', type=int, help='Resolve interval to use for all policies')
    args = parser.parse_args()

    run_experiment(config=args.config,
                   policies=args.policies,
                   iterations=args.iterations,
                   epsilon=args.epsilon,
                   resolve_interval=args.resolve_interval)
