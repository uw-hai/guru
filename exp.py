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
import random
import functools as ft
import copy
from pomdp import POMDPModel
from policy import Policy
from history import History
from simulator import Simulator, LiveSimulator
import util
from util import get_or_default, ensure_dir, equation_safe_filename
import analyze
import pymongo
import work_learn_problem as wlp
import hcomp_data_analyze.analyze

BOOTS_TERM = 5  # Terminate after booting this many workers in a row.

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

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
                   iteration=None, worker=None, policy=None):
    """Convert params to list of dictionaries to write to models file"""
    rows = []
    row_base = {'iteration': iteration,
                'worker': worker,
                'policy': policy}
    for p in params:
        row = {'param': p, 'v': params[p]}
        if hparams is not None:
            row['hyper'] = hparams[p]
        else:
            row['hyper'] = None
        row.update(row_base)
        rows.append(row)
    return rows

def run_function_from_dictionary(f, d):
    """Helper for Pool.map(), which can only use functions that take a single
    argument"""
    return f(**d)
 
def run_policy_iteration(exp_name, config_params, policy, iteration,
                         budget, budget_reserved_frac):
    """

    Seeds random number generators based on iteration only.

    Args:
        exp_name (str):                 Experiment name, without file ending.
        config_params (dict):           Params portion of config
        policy (dict):
        iteration (int):
        budget (float):
        budget_reserved_frac (float):   Fraction of budget reserved for
                                        exploitation.

    Returns:
        tuple:
            - Experiment rows to be written
            - Model rows to be written
    """
    # Seed iteration based on iteration only.
    np.random.seed(iteration)
    random.seed(iteration)

    it = iteration

    # Parse config
    params_gt = config_params
    n_worker_classes = len(config_params['p_worker'])

    pol = Policy(policy_type=policy['type'], exp_name=exp_name,
                 n_worker_classes=n_worker_classes, params_gt=params_gt,
                 **policy)

    # Begin experiment
    if 'dataset' in params_gt and params_gt['dataset'] is not None:
        simulator = LiveSimulator(params_gt, params_gt['dataset'])
    else:
        simulator = Simulator(params_gt)
    results = []
    history = History()

    budget_spent = 0
    worker_n = 0
    n_actions_by_worker = []
    t = 0
    reserved = False
    budget_explore = budget * (1 - budget_reserved_frac)
    while (budget_spent < budget and
           not (worker_n > BOOTS_TERM and
                all(n == 1 for n in n_actions_by_worker[-1 * BOOTS_TERM:])) and
           simulator.worker_available()):
        logger.info('{} (i:{}, w:{}, b:{:.2f}/{:.2f})'.format(
            pol, it, worker_n, budget_spent, budget))
        history.new_worker()
        s = simulator.new_worker()

        if budget_spent >= budget_explore:
            reserved = True

        # Belief using estimated model.
        pol.estimate_and_solve(iteration, history)
        belief = pol.model.get_start_belief()
        results.append({'iteration': it,
                        'worker': worker_n,
                        't': t,
                        'policy': str(pol),
                        'sys_t': time.clock(),
                        'a': None,
                        'explore': None,
                        'reserved': reserved,
                        's': s,
                        'o': None,
                        'cost': None,
                        'r': None,
                        'b': list(belief),
                        'other': None})
        worker_first_t = t
        t += 1

        while (budget_spent < budget and simulator.worker_hired()):
            if reserved:
                a = pol.get_best_action(it, history, belief)
                explore = False
            else:
                a, explore = pol.get_next_action(it, history, budget_spent,
                                                 budget_explore, belief)
            # Override policy decision and boot worker if in
            # entered reserved portion while worker hired.
            if not reserved and budget_spent >= budget_explore:
                a = pol.model.actions.index(wlp.Action('boot'))

            # Simulate a step
            s, o, (cost, r), other = simulator.sample_SOR(a)
            budget_spent -= cost
            history.record(a, o, explore=explore)
            belief = pol.model.update_belief(belief, a, o)

            results.append({'iteration': it,
                            'worker': worker_n,
                            't': t,
                            'policy': str(pol),
                            'sys_t': time.clock(),
                            'a': a,
                            'explore': explore,
                            's': s,
                            'o': o,
                            'cost': cost,
                            'r': r,
                            'b': list(belief),
                            'other': other})
            t += 1

        n_actions_by_worker.append(t - worker_first_t - 1)
        worker_n += 1

    # Record models, estimate times, and resolve times.
    models = []
    for worker in sorted(pol.params_estimated):
        params = pol.params_estimated[worker]
        if worker in pol.hparams_estimated:
            hparams = pol.hparams_estimated[worker]
        else:
            hparams = None
        models += params_to_rows(params=params,
                                 hparams=hparams,
                                 iteration=iteration,
                                 worker=worker,
                                 policy=str(pol))

    timings = []
    for worker in sorted(pol.estimate_times):
        timings.append({'iteration': iteration,
                        'worker': worker,
                        'policy': str(pol),
                        'type': 'estimate',
                        'duration': pol.estimate_times[worker]})

    for worker in sorted(pol.resolve_times):
        timings.append({'iteration': iteration,
                        'worker': worker,
                        'policy': str(pol),
                        'type': 'resolve',
                        'duration': pol.resolve_times[worker]})

    return results, models, timings

def run_experiment(name, mongo, config, policies, iterations, budget,
                   budget_reserved_frac,
                   epsilon=None, explore_actions=['test'], explore_policy=None,
                   thompson=False,
                   resolve_interval=None, hyperparams='HyperParams'):
    """Run experiment using multiprocessing.

    Args:
        name (str):             Name of experiment (config name).
        mongo (dict):           Connection details for mongo database.
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
        budget (float):         Maximum budget to spend before halting.
        budget_reserved_frac:   Fraction of budget reserved for exploitation.
        epsilon (str):          Exploration function string, with arguments
                                w (worker) and t (timestep).
        explore_actions (list): Action types for exploration.
        explore_policy (str):   Policy type name to use for exploration.
        thompson (bool):        Perform Thompson sampling.
        resolve_interval (int): Number of workers to see before resolving.
        hyperparams (str):      Hyperparams classname.

    """
    client = pymongo.MongoClient(mongo['host'], mongo['port'])
    client.worklearn.authenticate(mongo['user'], mongo['pass'],
                                  mechanism='SCRAM-SHA-1')
    exp_name = name
    models_path = os.path.join('models', exp_name)
    policies_path = os.path.join('policies', exp_name)
    for d in [models_path, policies_path]:
        ensure_dir(d)

    # If config already present, use that instead of passed configs.
    try:
        config = client.worklearn.config.find({'experiment': exp_name},
                                              {'_id': False,
                                               'experiment': False}).next()
    except StopIteration:
        config_insert = copy.deepcopy(config)
        config_insert['experiment'] = exp_name
        client.worklearn.config.insert(config_insert)
    params_gt = cmd_config_to_pomdp_params(config)
    n_worker_classes = len(params_gt['p_worker'])

    if explore_policy is not None:
        if epsilon is None:
            raise Exception('Must specify epsilon for explore_policy')
        matching_explore_policies = [
            p for p in policies if p['type'] == explore_policy]
        other_policies = [
            p for p in policies if p['type'] != explore_policy]
        # TODO: Check none of matching policies can be exploded.
        assert len(matching_explore_policies) == 1
        explore_policy = matching_explore_policies[0]
        policies = other_policies

    # Augment policies with exploration options.
    for p in policies:
        p['hyperparams'] = hyperparams
        if epsilon is not None:
            p['epsilon'] = epsilon
            p['explore_actions'] = explore_actions
            p['explore_policy'] = explore_policy
        if thompson:
            p['thompson'] = True
        if resolve_interval is not None:
            p['resolve_interval'] = resolve_interval

    # Explode policies.
    policies_exploded = []
    allowed_list_parameters = ['explore_actions']
    def flatten_single(p):
        for k in p:
            if (k not in allowed_list_parameters and
                    isinstance(p[k], list) and len(p[k]) == 1):
                p[k] = p[k][0]
    for p in policies:
        flatten_single(p)
        if 'explore_policy' in p and p['explore_policy'] is not None:
            flatten_single(p['explore_policy'])
        list_parameters = [
            k for k in p if
            k not in allowed_list_parameters and isinstance(p[k], list)]
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
        ensure_dir(os.path.join(models_path, str(i)))
        ensure_dir(os.path.join(policies_path, str(i)))

    # Prepare worker process arguments
    args_iter = ({'exp_name': exp_name,
                  'config_params': params_gt,
                  'policy': p,
                  'iteration': i,
                  'budget': budget,
                  'budget_reserved_frac': budget_reserved_frac} for i, p in
                 itertools.product(xrange(iterations),
                                   policies_exploded))

    # Write one-time rows.
    model_gt = POMDPModel(n_worker_classes, params=params_gt)
    if not list(client.worklearn.names.find({'experiment': exp_name})):
        for row in model_gt.get_names():
            row['experiment'] = exp_name
            client.worklearn.names.insert(row)
    if not list(client.worklearn.model.find({'experiment': exp_name})):
        for row in params_to_rows(model_gt.get_params_est()):
            row['experiment'] = exp_name
            row['param'] = str(row['param'])
            client.worklearn.model.insert(row)

    # Create worker processes.
    pool = mp.Pool(processes=util.cpu_count(), initializer=util.init_worker)
    f = ft.partial(util.run_functor, ft.partial(run_function_from_dictionary,
                                                run_policy_iteration))
    try:
        for res in pool.imap_unordered(f, args_iter):
            results_rows, models_rows, timings_rows = res
            for row in results_rows + models_rows + timings_rows:
                row['experiment'] = exp_name
            for row in models_rows:
                row['param'] = str(row['param'])

            # Delete any existing rows for this policy iteration.
            iteration = results_rows[0]['iteration']
            policy = results_rows[0]['policy']
            policy_iteration_query = {'experiment': exp_name,
                                      'iteration': iteration,
                                      'policy': policy}
            res_removed = client.worklearn.res.remove(
                policy_iteration_query)
            if res_removed['n'] > 0:
                print 'Removed {} result rows'.format(res_removed['n'])
            model_removed = client.worklearn.model.remove(
                policy_iteration_query)
            if model_removed['n'] > 0:
                print 'Removed {} result rows'.format(model_removed['n'])
            timing_removed = client.worklearn.timing.remove(
                policy_iteration_query)
            if timing_removed['n'] > 0:
                print 'Removed {} timing rows'.format(timing_removed['n'])

            # Store
            if results_rows:
                client.worklearn.res.insert(results_rows)
            if models_rows:
                client.worklearn.model.insert(models_rows)
            if timings_rows:
                client.worklearn.timing.insert(timings_rows)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.warn('Control-C pressed')
        pool.terminate()
    finally:
        pass
        # Plot.
        analyze.make_plots(
            db=client.worklearn,
            experiment=exp_name)

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
    for k in res.keys():
        if (k in ['p_learn_exp', 'p_learn_tell', 'p_lose',
                  'p_leave', 'p_slip', 'p_guess'] or
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
    parser.add_argument('--config_json', type=argparse.FileType('r'))
    config_group = parser.add_argument_group('config')
    config_group.add_argument(
        '--dataset', type=str, choices=[
        'lin_aaai12_tag', 'lin_aaai12_wiki', 'rajpal_icml15'],
        help='Dataset to use.')
    config_group.add_argument(
        '--p_worker', type=float, nargs='+', default=[1.0],
        help='Prior probabilities of worker classes')
    config_group.add_argument('--tell', dest='tell', action='store_true',
                              help="Allow 'tell' actions")
    config_group.add_argument('--exp',  dest='exp', action='store_true',
                              help="Allow 'exp(lain)' actions")
    config_group.add_argument('--cost', type=float, default=-0.1,
                              help="Cost of 'ask' actions.")
    config_group.add_argument('--cost_exp', type=float, default=-0.1,
                              help="Cost of 'exp(lain)' actions.")
    config_group.add_argument('--cost_tell', type=float, default=-0.1,
                              help="Cost of 'tell' actions.")
    config_group.add_argument('--p_learn_exp', type=float, nargs='+',
                              default=[0.4])
    config_group.add_argument('--p_learn_tell', type=float, nargs='+',
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
                              choices=['acc', 'pen'], default='pen')
    config_group.add_argument('--penalty_fp', type=float, default=-2)
    config_group.add_argument('--penalty_fn', type=float, default=-2)

    parser.add_argument('--policies', '-p', type=str, nargs='+', required=True,
                        choices=['teach_first', 'test_and_boot',
                                 'zmdp', 'appl', 'aitoolbox'])
    parser.add_argument('--explore_policy', type=str,
                        choices=['teach_first', 'test_and_boot'],
                        help='Use one of the baseline policies as the exploration policy')
    parser.add_argument('--teach_first_n', type=int, nargs='+')
    parser.add_argument('--teach_first_type', type=str,
                        choices=['tell', 'exp'], default='tell')
    parser.add_argument('--test_and_boot_n_test', type=int, nargs='+')
    parser.add_argument('--test_and_boot_n_work', type=int, nargs='+')
    parser.add_argument('--test_and_boot_accuracy', type=float, nargs='+')
    parser.add_argument('--test_and_boot_n_blocks', type=int,
                        help='Number of test-work blocks')
    parser.add_argument('--test_and_boot_final_action', type=str,
                        choices=['work', 'boot'], default='work',
                        help='Action to take after n test-work blocks')
    parser.add_argument('--zmdp_discount', type=float, nargs='+',
                        default=[0.99])
    parser.add_argument('--zmdp_timeout', type=int, nargs='+', default=[60])
    parser.add_argument('--appl_discount', type=float, nargs='+',
                        default=[0.99])
    parser.add_argument('--appl_timeout', type=int, nargs='+',
                        default=[60])
    parser.add_argument('--aitoolbox_discount', type=float, nargs='+',
                        default=[0.99])
    parser.add_argument('--aitoolbox_horizon', type=int, nargs='+')

    parser.add_argument('--iterations', '-i', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--budget', '-b', type=float, help='Total budget')
    parser.add_argument('--budget_reserved_frac', type=float, default=0.1,
                        help='Fraction of budget reserved for exploitation.')
    parser.add_argument('--epsilon', type=str,
                        help='Epsilon to use for all policies')
    parser.add_argument('--explore_actions', type=str, nargs='+',
                        choices=['test', 'work', 'tell', 'exp', 'boot'],
                        default=['test', 'work'])
    parser.add_argument(
        '--hyperparams', type=str, default='HyperParams',
        choices=['HyperParams', 'HyperParamsSpaced', 'HyperParamsWorker5',
                 'HyperParamsUnknownRatioWorker5',
                 'HyperParamsUnknownRatio',
                 'HyperParamsUnknownRatioSlipLeave',
                 'HyperParamsUnknownRatioSlipLeaveLose',
                 'HyperParamsSpacedUnknownRatio',
                 'HyperParamsSpacedUnknownRatioSlipLeave',
                 'HyperParamsSpacedUnknownRatioSlipLeaveLose'],
        help='Hyperparams class name, in param.py')
    parser.add_argument('--thompson', dest='thompson', action='store_true',
                        help="Use Thompson sampling")
    parser.add_argument('--resolve_interval', type=int, help='Resolve interval to use for all policies')
    args = parser.parse_args()
    args_vars = vars(args)

    if args.config_json is not None:
        config = json.load(args.config_json)
    else:
        config_params = [
            'p_worker', 'exp', 'tell', 'cost', 'cost_exp', 'cost_tell',
            'p_lose', 'p_leave',
            'p_slip', 'p_guess', 'p_r', 'p_1', 'p_s', 'utility_type',
            'dataset']
        if args.exp:
            config_params.append('p_learn_exp')
        if args.tell:
            config_params.append('p_learn_tell')
        if args.utility_type == 'pen':
            config.params += ['penalty_fp', 'penalty_fn']
        config = dict((k, args_vars[k]) for k in config_params)

    # For live datasets, default budget to cost of asking all questions.
    if ('dataset' in config and
            config['dataset'] is not None and
            args.budget is None):
        if config['dataset'] == 'lin_aaai12_tag':
            data = hcomp_data_analyze.analyze.Data.from_lin_aaai12(
                workflow='tag')
        elif config['dataset'] == 'lin_aaai12_wiki':
            data = hcomp_data_analyze.analyze.Data.from_lin_aaai12(
                workflow='wiki')
        elif config['dataset'] == 'rajpal_icml15':
            data = hcomp_data_analyze.analyze.Data.from_rajpal_icml15(
                worker_type=None)
        args.budget = config['cost'] * data.get_n_answers()

    policies = []
    for p_type in args.policies:
        p = {'type': p_type}
        if p_type == 'teach_first':
            p['n'] = args.teach_first_n
            p['teach_type'] = args.teach_first_type
        elif p_type == 'test_and_boot':
            p['n_test'] = args.test_and_boot_n_test
            p['n_work'] = args.test_and_boot_n_work
            p['accuracy'] = args.test_and_boot_accuracy
            p['n_blocks'] = args.test_and_boot_n_blocks
            p['final_action'] = args.test_and_boot_final_action
        elif p_type == 'zmdp':
            p['discount'] = args.zmdp_discount
            p['timeout'] = args.zmdp_timeout
        elif p_type == 'appl':
            p['discount'] = args.appl_discount
            p['timeout'] = args.appl_timeout
        elif p_type == 'aitoolbox':
            p['discount'] = args.aitoolbox_discount
            p['horizon'] = args.aitoolbox_horizon
        policies.append(p)

    run_experiment(name=args.name,
                   mongo={'host': os.environ['MONGO_HOST'],
                          'port': int(os.environ['MONGO_PORT']),
                          'user': os.environ['MONGO_USER'],
                          'pass': os.environ['MONGO_PASS']},
                   config=config,
                   policies=policies,
                   iterations=args.iterations,
                   budget=args.budget,
                   budget_reserved_frac=args.budget_reserved_frac,
                   epsilon=args.epsilon,
                   explore_actions=args.explore_actions,
                   explore_policy=args.explore_policy,
                   thompson=args.thompson,
                   resolve_interval=args.resolve_interval,
                   hyperparams=args.hyperparams)
