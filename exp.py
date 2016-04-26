from __future__ import division
import multiprocessing as mp
import argparse
import itertools
import os
import time
import json
import logging
import random
import functools as ft
import copy
import numpy as np
import pymongo
from . import param
from .pomdp import POMDPModel
from .policy import Policy
from .history import History
from .simulator import Simulator, LiveSimulator
from . import util
from .util import ensure_dir
from . import analyze
from . import work_learn_problem as wlp
from .hcomp_data_analyze import analyze as hanalyze

BOOTS_TERM = 5  # Terminate after booting this many workers in a row.

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)


def parseNumList(string):
    import re
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError(
            "'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start, 10), int(end, 10) + 1))


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


def run_policy_iteration(exp_name, params_gt, params_policy, policy, iteration,
                         budget, budget_reserved_frac, passive):
    """

    Seeds random number generators based on iteration only.

    Args:
        exp_name (str):                 Experiment name, without file ending.
        params_gt (Params object):      Params portion of config.
        params_policy (Params object):  Policy params.
        policy (dict):
        iteration (int):
        budget (float):
        budget_reserved_frac (float):   Fraction of budget reserved for
                                        exploitation.
        passive (bool):                 Passive learning.

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
    if params_policy is None:
        params_policy = params_gt
    n_worker_classes = params_policy.n_classes

    pol = Policy(policy_type=policy['type'],
                 n_worker_classes=n_worker_classes,
                 params_gt=params_policy.get_param_dict(sample=False),
                 **policy)

    # Begin experiment
    # TODO: Don't make assumptions about random workers / actions.
    if 'dataset' in params_gt.params and params_gt.params['dataset']:
        simulator = LiveSimulator(
            params_gt, repeat=True, random_workers=True, random_actions=True)
    else:
        simulator = Simulator(params_gt)
    if passive:
        passive_simulator = LiveSimulator(
            params_gt, repeat=False,
            random_workers=False, random_actions=False)
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
           simulator.worker_available() and
           (not passive or passive_simulator.worker_available())):
        # BUG: Line above means passive always stops before running another
        # simulator.
        logger.info('{} (i:{}, w:{}, b:{:.2f}/{:.2f})'.format(
            pol, it, worker_n, budget_spent, budget))
        if passive and passive_simulator.worker_available():
            curr_simulator = passive_simulator
            using_passive = True
        else:
            curr_simulator = simulator
            using_passive = False
        s = curr_simulator.new_worker()

        if budget_spent >= budget_explore:
            reserved = True

        # Belief using estimated model.
        resolve_min_worker_interval = 1 if using_passive else 10
        resolve_max_n = None if using_passive else 10
        logger.info('prepping worker...')

        # Prep for worker.
        pomdp_dirpath = os.path.join(
            os.path.dirname(__file__), 'models', exp_name, str(it))
        policy_dirpath = os.path.join(
            os.path.dirname(__file__), 'policies', exp_name, str(it))
        pomdp_fpath = os.path.join(
            pomdp_dirpath, '{}-{:06d}.pomdp'.format(str(pol), worker_n))
        policy_fpath = os.path.join(
            policy_dirpath, '{}-{:06d}.policy'.format(str(pol), worker_n))
        pol.prep_worker(
            model_filepath=pomdp_fpath,
            policy_filepath=policy_fpath,
            history=history,
            budget_spent=budget_spent,
            budget_explore=budget_explore,
            reserved=reserved,
            resolve_min_worker_interval=resolve_min_worker_interval,
            resolve_max_n=resolve_max_n)
        history.new_worker()
        logger.info('...prepped')
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

        while (budget_spent < budget and curr_simulator.worker_hired()):
            if not using_passive:
                if reserved:
                    a = pol.get_best_action(history=history, belief=belief)
                    explore = False
                else:
                    a, explore = pol.get_next_action(history, budget_spent,
                                                     budget_explore, belief)
                # Override policy decision and boot worker if in
                # entered reserved portion while worker hired.
                if not reserved and budget_spent >= budget_explore:
                    a = pol.model.actions.index(wlp.Action('boot'))
            else:
                explore = False

            # Simulate a step
            if not using_passive:
                a, s, o, (cost, r), other = simulator.sample_SOR(a=a)
            else:
                a, s, o, (cost, r), other = passive_simulator.sample_SOR(a=None)
                s = None
                # TODO: Record whether following passive or not.
            budget_spent -= cost
            logger.info('{} (i:{}, w:{}, a:{}, o:{}, b:{:.2f}/{:.2f})'.format(
                pol, it, worker_n, a, o, budget_spent, budget))
            history.record(a, o, explore=explore)
            belief = pol.model.update_belief(belief, a, o)

            results.append({'iteration': it,
                            'worker': worker_n,
                            't': t,
                            'policy': str(pol),
                            'sys_t': time.clock(),
                            'a': a,
                            'explore': explore,
                            'reserved': reserved,
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


def run_experiment(name, mongo, config, config_policy,
                   policies, iterations, budget,
                   budget_reserved_frac,
                   epsilon=None, explore_actions=['test'], explore_policy=None,
                   thompson=False, hyperparams='HyperParams', processes=None,
                   passive=False):
    """Run experiment using multiprocessing.

    Args:
        name (str):             Name of experiment (config name).
        mongo (dict):           Connection details for mongo database.
        config (dict):          Config dictionary, in format expected by
                                POMDPModel. If experiment folder
                                already exists and contains config.json,
                                ignore this parameter and use that instead.
        config_policy (dict):   Params for _policy_, in format expected by
                                POMDPModel.
        policies (list):        List of policy dictionaries. Acceptable
                                to use compressed format where multiple
                                policies can be represented in a single
                                dictionary by substituting a single
                                parameter value with a list.
        iterations (list):      List of iterations.
        budget (float):         Maximum budget to spend before halting.
        budget_reserved_frac:   Fraction of budget reserved for exploitation.
        epsilon (str):          Exploration function string, with arguments
                                w (worker) and t (timestep).
        explore_actions (list): Action types for exploration.
        explore_policy (str):   Policy type name to use for exploration.
        thompson (bool):        Perform Thompson sampling.
        hyperparams (str):      Hyperparams classname.
        processes (int):        Number of processes.
        passive (bool):         Passive learning mode.

    """
    client = pymongo.MongoClient(mongo['host'], mongo['port'])
    if mongo['user']:
        client.worklearn.authenticate(mongo['user'], mongo['pass'],
                                      mechanism='SCRAM-SHA-1')
    exp_name = name
    models_path = os.path.join(os.path.dirname(__file__), 'models', exp_name)
    policies_path = os.path.join(
        os.path.dirname(__file__), 'policies', exp_name)
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
    params_gt = param.Params.from_cmd(config)
    if config_policy is not None:
        params_policy = param.Params.from_cmd(config_policy)
    else:
        params_policy = None

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
            raise Exception(
                'Policies must contain only a single list parameter')

    # Make folders (errors when too many folders are made in subprocesses).
    for i in iterations:
        ensure_dir(os.path.join(models_path, str(i)))
        ensure_dir(os.path.join(policies_path, str(i)))

    # Prepare worker process arguments
    args_iter = ({'exp_name': exp_name,
                  'params_gt': params_gt,
                  'params_policy': params_policy,
                  'policy': p,
                  'iteration': i,
                  'budget': budget,
                  'budget_reserved_frac': budget_reserved_frac,
                  'passive': passive} for i, p in
                 itertools.product(iterations,
                                   policies_exploded))

    # Write one-time rows.
    n_worker_classes = params_gt.n_classes
    model_gt = POMDPModel(
        n_worker_classes,
        params=params_gt.get_param_dict(sample=False))
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
    nprocesses = processes or util.cpu_count()
    pool = mp.Pool(processes=nprocesses, initializer=util.init_worker)
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
            experiment=exp_name,
            processes=nprocesses)


def add_config_argparse_group(parser):
    """Add config argument group to parser.

    Args:
        parser: ArgParse parser.

    Returns:
        config_group

    """
    config_group = parser.add_argument_group('config')
    config_group.add_argument(
        '--passive', dest='passive', action='store_true',
        help='Use passive learning mode only')
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
    config_group.add_argument(
        '--p_learn_exp', type=float, nargs='+', default=[0.4],
        help='Probability of learning a rule from an explain action. Index changes by rule, then worker class (or just rule if shared across worker classes).')
    config_group.add_argument(
        '--p_learn_tell', type=float, nargs='+', default=[0.4],
        help='Probability of learning a rule from a tell action. Index changes by rule, then worker class (or just rule if shared across worker classes).')
    config_group.add_argument(
        '--p_lose', type=float, nargs='+', default=[0],
        help='Probability of losing knowledge of a rule. Index changes by rule, then worker class (or just rule if shared across worker classes).')
    config_group.add_argument('--p_leave', type=float, nargs='+',
                              default=[0.01])
    config_group.add_argument(
        '--p_slip', type=float, nargs='+', default=[0.1],
        help='Probability of "slipping" (answering incorrectly) when rules are known. Index changes by question type, then worker class (or just question type if shared across worker classes).')
    config_group.add_argument(
        '--p_slip_std', type=float, nargs='+', default=[0],
        help='Standard deviation for simulations')
    config_group.add_argument(
        '--p_guess', type=float, nargs='+', default=[0.5],
        help='Probability of guessing correctly when rules are not known. Index changes by question type, then worker class (or just question type if shared across worker classes).')
    config_group.add_argument(
        '--p_r', type=float, nargs='+', default=[1],
        help='Prior probability rule needed to answer question.')
    config_group.add_argument('--p_1', type=float, nargs='+', default=[0.5],
        help='Prior probability "1" is correct answer (for boolean questions). One value for each question type.')
    config_group.add_argument(
        '--p_s', type=float, nargs='+', default=[0.2],
        help='Probability worker initially knows a rule. Index changes by rule, then worker class (or just rule if shared across worker classes).')
    config_group.add_argument('--utility_type', type=str,
                              choices=['acc', 'pen',
                                       'pen_diff', 'pen_nonboolean'],
                              default='pen')
    config_group.add_argument('--penalty_fp', type=float, default=-2)
    config_group.add_argument('--penalty_fn', type=float, default=-2)
    config_group.add_argument('--reward_tp', type=float, default=1)
    config_group.add_argument('--reward_tn', type=float, default=1)
    config_group.add_argument(
        '--desired_accuracy', type=float,
        help='Desired accuracy for utility_types "pen" and "pen_nonboolean", which overrides penalty and reward settings')
    config_group.add_argument(
        '--desired_accuracy_rewards', dest='desired_accuracy_rewards',
        action='store_true', help='Overwrite penalty & reward')
    return config_group


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('name', type=str, help='Experiment name')
    parser.add_argument('--config_json', type=argparse.FileType('r'))
    parser.add_argument('--proc', type=int, help='Number of processes')
    add_config_argparse_group(parser)
    parser.add_argument('--policies', '-p', type=str, nargs='+', required=True,
                        choices=['work_only', 'test_and_boot',
                                 'zmdp', 'appl', 'aitoolbox'])
    parser.add_argument('--explore_policy', type=str,
                        choices=['test_and_boot'],
                        help='Use one of the baseline policies as the exploration policy')
    parser.add_argument('--accuracy_bins_n', type=int,
                        help='Number of accuracy bins (classes) to use for the policy model.')
    parser.add_argument('--teach_type', type=str, choices=['tell', 'exp'])
    parser.add_argument('--test_and_boot_n_teach', type=int, nargs='+')
    parser.add_argument('--test_and_boot_n_test', type=int, nargs='+')
    parser.add_argument('--test_and_boot_n_work', type=int, nargs='+')
    parser.add_argument('--test_and_boot_accuracy', type=float, nargs='+')
    parser.add_argument('--test_and_boot_accuracy_window',
                        type=float, nargs='+')
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

    parser.add_argument('--iterations', '-i', type=parseNumList,
                        default=range(100), help='Iterations')
    parser.add_argument('--budget', '-b', type=float, help='Total budget')
    parser.add_argument('--budget_reserved_frac', type=float, default=0.0,
                        help='Fraction of budget reserved for exploitation.')
    parser.add_argument('--epsilon', type=str, nargs='*',
                        help='Epsilon to use for all policies')
    parser.add_argument('--explore_actions', type=str, nargs='+',
                        choices=['test', 'work', 'tell', 'exp', 'boot'],
                        default=['test', 'work'])
    parser.add_argument(
        '--hyperparams', type=str, default='HyperParams',
        choices=param.HYPERPARAMS, help='Hyperparams class name, in param.py')
    parser.add_argument('--thompson', dest='thompson', action='store_true',
                        help="Use Thompson sampling")
    args = parser.parse_args()
    args_vars = vars(args)

    config_params = [
        'p_worker', 'exp', 'tell', 'cost', 'cost_exp', 'cost_tell',
        'p_lose', 'p_leave',
        'p_slip', 'p_slip_std', 'p_guess', 'p_r', 'p_1', 'p_s',
        'utility_type', 'dataset']
    if args.exp:
        config_params.append('p_learn_exp')
    if args.tell:
        config_params.append('p_learn_tell')
    if args.utility_type in ['pen', 'pen_diff', 'pen_nonboolean']:
        config_params += ['penalty_fp', 'penalty_fn', 'reward_tp', 'reward_tn']
    if args.dataset is None:
        config_params.append('p_slip_std')

    if args.config_json is not None:
        config = json.load(args.config_json)
    else:
        config = dict()
    for k in config_params:
        if k not in config:
            config[k] = args_vars[k]

    # Overwrite reward settings for desired accuracy.
    if (config['utility_type'] in ['pen', 'pen_nonboolean'] and
            args.desired_accuracy is not None and
            args.desired_accuracy_rewards):
        config['reward_tp'] = 1
        config['reward_tn'] = 1
        p = util.get_penalty(args.desired_accuracy, reward=1)
        config['penalty_fp'] = p
        config['penalty_fn'] = p

    if args.accuracy_bins_n == 2 and args.desired_accuracy is not None:
        config_policy = dict()
        config_policy = copy.deepcopy(config)
        config_policy['p_worker'] = [0.5, 0.5]  # BUG: Why did we do this?
        # BUG: Fixing the class accuracies at midpoints seems stupid,
        # but let's do anyways.
        p_slip_thresh = 1 - args.desired_accuracy
        config_policy['p_slip'] = [
            p_slip_thresh / 2, (p_slip_thresh + 0.5) / 2]
        config_policy['desired_accuracy'] = args.desired_accuracy
    elif args.accuracy_bins_n is not None:
        # BUG: Why separate case?
        n = args.accuracy_bins_n
        config_policy = dict()
        config_policy = copy.deepcopy(config)
        config_policy['p_worker'] = [1 / n for i in xrange(n)]
        config_policy['p_slip'] = util.midpoints(0.0, 0.5, n)
        # BUG: Store desired accuracy in config_policy?
    else:
        config_policy = None

    # For live datasets, default budget to cost of asking all questions.
    if ('dataset' in config and
            config['dataset'] is not None and
            args.budget is None):
        if config['dataset'] == 'lin_aaai12_tag':
            data = hanalyze.Data.from_lin_aaai12(
                workflow='tag')
        elif config['dataset'] == 'lin_aaai12_wiki':
            data = hanalyze.Data.from_lin_aaai12(
                workflow='wiki')
        elif config['dataset'] == 'rajpal_icml15':
            data = hanalyze.Data.from_rajpal_icml15(
                worker_type=None)
        args.budget = -1 * config['cost'] * data.get_n_answers()

    policies = []
    for p_type in args.policies:
        p = {'type': p_type}
        if p_type == 'test_and_boot':
            p['n_test'] = args.test_and_boot_n_test
            p['n_teach'] = args.test_and_boot_n_teach
            p['teach_type'] = args.teach_type
            p['n_work'] = args.test_and_boot_n_work
            p['accuracy'] = args.test_and_boot_accuracy
            p['accuracy_window'] = args.test_and_boot_accuracy_window
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

    try:
        epsilon = ' '.join(args.epsilon) or None
    except TypeError:
        epsilon = None
    run_experiment(name=args.name,
                   mongo={
                       'host': os.environ['MONGO_HOST'],
                       'port': int(os.environ['MONGO_PORT']),
                       'user': os.environ.get('MONGO_USER', None),
                       'pass': os.environ.get('MONGO_PASS', None)},
                   config=config,
                   config_policy=config_policy,
                   policies=policies,
                   iterations=args.iterations,
                   budget=args.budget,
                   budget_reserved_frac=args.budget_reserved_frac,
                   epsilon=epsilon,
                   explore_actions=args.explore_actions,
                   explore_policy=args.explore_policy,
                   thompson=args.thompson,
                   hyperparams=args.hyperparams,
                   processes=args.proc,
                   passive=args.passive)
