import unittest
import json
import os
from .. import pomdp
from .. import policy


def get_external_policy(exp_name, params, policy_params):
    p = policy.Policy(exp_name=exp_name, **policy_params)
    external_policy = p.run_solver(
        model_filename=os.path.join(os.path.dirname(__file__), 'tmp', '{}.pomdp'.format(exp_name)),
        policy_filename=os.path.join(os.path.dirname(__file__), 'tmp', '{}.policy'.format(exp_name)),
        params=params)
    return external_policy



class TestAITPolicy(unittest.TestCase):
    def setUp(self):
        pass

    def testBeliefs2sk(self):
        """

        beliefs = [[0, 0.2, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]


        with open('test/data/2sk-acc-p_s0.2.json', 'r') as f:
            params = json.load(f)
        model = pomdp.POMDPModel(**params)
        p = get_external_policy('2sk-acc-p_s0.2', params,
                                {'policy_type': 'aitoolbox', 'horizon': 50})
        for b in beliefs:
            for s,v in zip(model.states, b):
                if v > 0:
                    print '{}: {}'.format(s,v)
            print p.get_action_rewards(b)
        """

    def testBeliefs1sk(self):
        """

        # ['TERM', 's1', 's0', 's1q0', 's0q0']
        beliefs1sk = [[0, 0, 1, 0, 0],
                      [0, 0.2, 0.8, 0, 0],
                      [0, 0.8, 0.2, 0, 0],
                      [0, 1.0, 0, 0, 0] ]

        with open('test/data/1sk-acc-p_s0.2.json', 'r') as f:
            params = json.load(f)
        model = pomdp.POMDPModel(**params)
        print [str(s) for s in model.states]
        print [str(s) for s in model.actions]
        p = get_external_policy('1sk-acc-p_s0.2', params,
                                {'policy_type': 'aitoolbox', 'horizon': 50})
        for b in beliefs1sk:
            for s,v in zip(model.states, b):
                if v > 0:
                    print '{}: {}'.format(s,v)
            print p.get_action_rewards(b)
        """



if __name__ == '__main__':
    unittest.main()
