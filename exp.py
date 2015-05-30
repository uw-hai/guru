import multiprocessing as mp
import argparse
import csv
import os
import time
import json
import numpy as np
from pomdp import POMDPModel
from policy import Policy
from history import History
from util import get_or_default, ensure_dir

def get_start_belief(fp):
    """DEPRECATED"""
    for line in fp:
        if line.strip().startswith('start:'):
            start_belief = np.array([float(s) for s in line[6:].split()])
            return start_belief
    return None

def belief_to_str(lst):
    return ' '.join(str(x) for x in lst)

def run_experiment(config_fp):
    basename = os.path.basename(config_fp.name)
    name_exp = os.path.splitext(basename)[0]

    # Parse config
    config = json.load(config_fp)
    params_gt = config['params']
    params_gt['utility_type'] = get_or_default(
        params_gt, 'utility_type', 'acc')
    params_gt['p_lose'] = get_or_default(params_gt, 'p_lose', 0)
    params_gt_fixed = dict((k, params_gt[k]) for k in
                           ['cost', 'cost_exp', 'p_r', 'p_1'])

    n_skills = len(params_gt['p_s'])
    iterations = get_or_default(params_gt, 'iterations', 1)
    episodes = params_gt['episodes']
    policies = [Policy(policy_type=p['type'],
                       exp_name=name_exp, **p) for p in config['policies']]


    # Always write a 0.99 discount ground truth .pomdp file.
    # TODO: We don't need this.
    ensure_dir(os.path.join('models', name_exp, 'gt'))
    model_gt = POMDPModel(n_skills=n_skills, **params_gt)
    model_gt_path = os.path.join('models', name_exp, 'gt', 'gt.pomdp')
    with open(model_gt_path, 'w') as f:
        model_gt.write_pomdp(f, discount=0.99)
    model_actions = model_gt.actions
    model_states = model_gt.states
    model_observations = model_gt.observations

    #ensure_dir(os.path.join('policies', name_exp))
    # TODO: Why doesn't this work without the name_exp part?
    ensure_dir(os.path.join('res', name_exp))

    #filename_policy = '../appl-0.96/src/{}.policy'.format(name_exp)
    #filename_env = '{}.pomdp'.format(name_exp)
    #pomdp = pp.POMDP(filename_env, filename_policy, start_belief)

    model_est_path= os.path.join('res', '{}_model.csv'.format(name_exp))
    with open(model_est_path, 'wb') as f:
        model_gt.write_params(f, writeheader=True)
    with open(os.path.join('res', '{}_names.csv'.format(name_exp)), 'wb') as f:
        model_gt.write_names(f)

    results_fields = ['iteration','episode','t','policy','sys_t',
                      'a','s','o','r','b']
    results_fp = open(os.path.join('res', '{}.txt'.format(name_exp)), 'wb')
    r_writer = csv.DictWriter(results_fp, fieldnames=results_fields)
    r_writer.writeheader()

    for it in xrange(iterations):
        for pol in policies:
            # TODO: Clear policy for new run.
            history = History()
            if pol.epsilon is not None:
                model_est = POMDPModel(n_skills=n_skills, **params_gt_fixed)
            else:
                model_est = model_gt

            for ep in xrange(episodes):
                if pol.epsilon is not None and ep % pol.estimate_interval == 0:
                    # TODO: Reestimate only for policies that use POMDP solvers?
                    first_ep = ep == 0
                    model_est.estimate(history, random_init=first_ep)
                params_est = model_est.params
                with open(model_est_path, 'ab') as f:
                    model_est.write_params(
                        f, writeheader=False,
                        iteration=it, episode=ep, policy=pol)

                history.new_episode()
                start_belief = model_gt.get_start_belief()
                start_state = np.random.choice(range(len(start_belief)),
                                               p=start_belief)
                s = start_state

                # Belief using estimated model.
                belief = model_est.get_start_belief()
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
                results_fp.flush()

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
