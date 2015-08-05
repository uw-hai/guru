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
import copy
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

def params_to_rows(params, hparams=None,
                   iteration=None, episode=None, policy=None):
    """Convert params to list of dictionaries to write to models file"""
    rows = []
    row_base = {'iteration': iteration,
                'episode': episode,
                'policy': policy}
    for p in params:
        row = {'param': p, 'v': params[p]}
        if hparams is not None:
            row['hyper'] = hparams[p]
        row.update(row_base)
        rows.append(row)
    return rows

def run_function_from_dictionary(f, d):
    """Helper for Pool.map(), which can only use functions that take a single
    argument"""
    return f(**d)
 
def run_policy_iteration(exp_name, config_params, policy, iteration, episodes):
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
    n_worker_classes = len(config_params['p_worker'])

    pol = Policy(policy_type=policy['type'], exp_name=exp_name, **policy)

    # GT model
    model_gt = POMDPModel(n_worker_classes, params=params_gt)

    results = []

    # Begin experiment
    history = History()
    if pol.rl_p():
        model_est = POMDPModel(n_worker_classes, params=params_gt,
                               estimate_all=True)
    else:
        model_est = model_gt

    for ep in xrange(episodes):
        logger.info('{} ({}, {})'.format(pol, it, ep))
        history.new_episode()
        start_belief = model_gt.get_start_belief()
        #logger.info('start belief: {}'.format(list(start_belief)))
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
            a = pol.get_next_action(model_est, it, history, valid_actions,
                                    belief)

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
        if ep in pol.hparams_estimated:
            hparams = pol.hparams_estimated[ep]
        else:
            hparams = None
        models += params_to_rows(params=params,
                                 hparams=hparams,
                                 iteration=iteration,
                                 episode=ep,
                                 policy=str(pol))

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

def run_experiment(name, config, policies, iterations, episodes, epsilon=None,
                   thompson=False, resolve_interval=None):
    """Run experiment using multiprocessing.

    Args:
        name:                   Name of experiment (config name).
        config (dict):          Config dictionary, in format expected by
                                POMDPModel. If experiment folder
                                already exists and contains config.json,
                                ignore this parameter and use that instead.
        policies (list):        List of policy dictionaries. Acceptable
                                to use compressed format where multiple
                                policies can be represented in a single
                                dictionary by substituting a single
                                parameter value with a list.
        iterations (int):       Number of iterations.
        episodes (int):         Number of episodes.
        epsilon (str):          Exploration function string, with arguments
                                e (episode) and t (timestep).
        thompson (bool):        Perform Thompson sampling.
        resolve_interval (int): Number of episodes before resolving.

    """
    exp_name = name
    if episodes > 1:
        exp_name += '-ep{}'.format(episodes)
    res_path = os.path.join('res', exp_name)
    models_path = os.path.join('models', exp_name)
    policies_path = os.path.join('policies', exp_name)
    for d in [res_path, models_path, policies_path]:
        ensure_dir(d)

    # If config file already present, use that instead of passed configs.
    config_path = os.path.join(res_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    params_gt = cmd_config_to_pomdp_params(config)
    n_worker_classes = len(params_gt['p_worker'])

    # Augment policies with exploration options.
    for p in policies:
        if epsilon is not None:
            p['epsilon'] = epsilon
        if thompson:
            p['thompson'] = True
        if resolve_interval is not None:
            p['resolve_interval'] = resolve_interval
    # Create aggregate name for policies.
    policies_name = ''
    for i, p in enumerate(policies):
        if i:
            policies_name += '-'
        policies_name += p['type']
        for k in (k for k in p if k not in ['type',
                                            'epsilon',
                                            'thompson',
                                            'resolve_interval']):
            if isinstance(p[k], list):
                value_string  = '_'.join(str(x) for x in p[k])
            else:
                value_string = p[k]
            policies_name += '-{}_{}'.format(k, value_string)
    if epsilon is not None:
        policies_name += '-eps_{}'.format(equation_safe_filename(epsilon))
    if thompson:
        policies_name += '-thomp'
    if resolve_interval is not None:
        policies_name += '-s_int_{}'.format(resolve_interval)

    # Explode policies.
    policies_exploded = []
    for p in policies:
        list_parameters = [k for k in p if isinstance(p[k], list)]
        if len(list_parameters) == 0:
            policies_exploded.append(p)
        elif len(list_parameters) == 1:
            k = list_parameters[0]
            for v in p[k]:
                p_prime = copy.deepcopy(p)
                p_prime[k] = v
                policies_exploded.append(p_prime)
        else:
            raise Exception('Policies must contain only a single list parameter')

    # Make folders (errors when too many folders are made in subprocesses).
    for i in xrange(iterations):
        for ep in xrange(episodes):
            ensure_dir(os.path.join(models_path, str(i), str(ep)))
            ensure_dir(os.path.join(policies_path, str(i), str(ep)))

    # Prepare worker process arguments
    args_iter = ({'exp_name': exp_name,
                  'config_params': params_gt,
                  'policy': p,
                  'iteration': i,
                  'episodes': episodes} for i, p in
                 itertools.product(xrange(iterations), policies_exploded))

    # Write one-time files.
    model_gt = POMDPModel(n_worker_classes, params=params_gt)
    with open(os.path.join(res_path, '{}_names.csv'.format(policies_name)), 'wb') as f:
        model_gt.write_names(f)

    # Open file pointers.
    models_fp = open(
        os.path.join(res_path, '{}_model.csv'.format(policies_name)), 'wb')
    models_fieldnames = ['iteration', 'episode', 'policy', 'param',
                         'v', 'hyper']
    m_writer = csv.DictWriter(models_fp, fieldnames=models_fieldnames)
    m_writer.writeheader()

    for r in params_to_rows(model_gt.get_params_est()):
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
    f = ft.partial(run_functor, ft.partial(run_function_from_dictionary,
                                           run_policy_iteration))
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

def cmd_config_to_pomdp_params(config):
    """Convert command line config parameters to params for POMDPModel.

    Notes:
    - 'p_worker' must give full categorical probability vector.
    - Other probabilities are bernoulli distributions and must be given only
      using positive probability.
    - Bernoulli distributions can either be conditioned on p_worker, or not.
    
    Infers whether Bernoulli distributions are conditioned or use parameter
    tying from the number of parameters specified.

    Args:
        config: Dictionary of command line config parameters.

    Returns:
        New dictionary of parameters.

    """
    n_worker_classes = len(config['p_worker'])
    n_rules = len(config['p_r'])

    # Copy dictionary and split p_s by rule.
    res = dict()
    for k in config:
        if k == 'p_s':
            if (len(config[k]) != n_rules and
                len(config[k]) != n_rules * n_worker_classes):
                raise Exception('Config input of unexpected size')
            for i, v in enumerate(config[k]):
                if i < n_rules:
                    res[k, i] = []
                res[k, i % n_rules].append(v)
        else:
            if (isinstance(config[k], list) and len(config[k]) > 1 and
                len(config[k]) != n_worker_classes):
                raise Exception('Config input of unexpected size')
            res[k] = config[k]

    # Make bernoulli probabilites full probabilities.
    # TODO: Move into POMDPModel?
    for k in res:
        if (k in ['p_learn', 'p_lose', 'p_leave', 'p_slip', 'p_guess'] or
            (len(k) == 2 and k[0] == 'p_s')):
            probs = res.pop(k)
            if len(probs) == 1:
                res[k, None] = [probs[0], 1 - probs[0]]
            else:
                for i, v in enumerate(probs):
                    res[k, i] = [v, 1 - v]
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('name', type=str, help='Experiment name')

    config_group = parser.add_argument_group('config')
    config_group.add_argument(
        '--p_worker', type=float, nargs='+', default=[1.0],
        help='Prior probabilities of worker classes')
    config_group.add_argument('--cost', type=float, default=-0.1)
    config_group.add_argument('--cost_exp', type=float, default=-0.1)
    config_group.add_argument('--p_learn', type=float, nargs='+',
                              default=[0.4])
    config_group.add_argument('--p_lose', type=float, nargs='+',
                              default=[0])
    config_group.add_argument('--p_leave', type=float, nargs='+',
                              default=[0.01])
    config_group.add_argument('--p_slip', type=float, nargs='+',
                              default=[0.1])
    config_group.add_argument('--p_guess', type=float, nargs='+',
                              default=[0.5])
    config_group.add_argument('--p_r', type=float, nargs='+', default=[0.5])
    config_group.add_argument('--p_1', type=float, default=0.5)
    config_group.add_argument('--p_s', type=float, nargs='+', default=[0.2])
    config_group.add_argument('--utility_type', type=str,
                              choices=['acc', 'posterior'], default='acc')

    parser.add_argument('--policies', '-p', type=str, nargs='+', required=True,
                        choices=['fixed', 'zmdp', 'appl', 'aitoolbox'])
    parser.add_argument('--fixed_n', type=int, nargs='+')
    parser.add_argument('--zmdp_discount', type=float, default=0.99)
    parser.add_argument('--zmdp_timeout', type=int, nargs='+', default=[60])
    parser.add_argument('--appl_discount', type=float, default=0.99)
    parser.add_argument('--appl_timeout', type=int, nargs='+', default=[60])
    parser.add_argument('--aitoolbox-discount', type=float, default=0.99)
    parser.add_argument('--aitoolbox-horizon', type=int, nargs='+')

    parser.add_argument('--iterations', '-i', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--episodes', '-e', type=int, default=1,
                        help='Number of episodes')
    parser.add_argument('--epsilon', type=str,
                        help='Epsilon to use for all policies')
    parser.add_argument('--thompson', dest='thompson', action='store_true',
                        help="Use Thompson sampling")
    parser.set_defaults(thompson=False)
    parser.add_argument('--resolve_interval', type=int, help='Resolve interval to use for all policies')
    args = parser.parse_args()
    args_vars = vars(args)

    config_params = [
        'p_worker', 'cost', 'cost_exp', 'p_learn', 'p_lose', 'p_leave',
        'p_slip', 'p_guess', 'p_r', 'p_1', 'p_s', 'utility_type']
    config = dict((k, args_vars[k]) for k in config_params if
                  args_vars[k] is not None)

    policies = []
    for p_type in args.policies:
        p = {'type': p_type}
        if p_type == 'fixed':
            p['n'] = args.fixed_n
        elif p_type == 'zmdp':
            p['discount'] = args.zmdp_discount
            p['timeout'] = args.zmdp_timeout
        elif p_type == 'appl':
            p['discount'] = args.appl_discount
            p['timeout'] = args.appl_timeout
        elif p_type == 'aitoolbox':
            p['discount'] = args.aitoolbox-discount
            p['horizon'] = args.aitoolbox-horizon
        policies.append(p)

    run_experiment(name=args.name,
                   config=config,
                   policies=policies,
                   iterations=args.iterations,
                   episodes=args.episodes,
                   epsilon=args.epsilon,
                   thompson=args.thompson,
                   resolve_interval=args.resolve_interval)
