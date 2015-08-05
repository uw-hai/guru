import unittest

import pomdp
#import worklearn
#from worklearn import pomdp

class TestPolicyImport(unittest.TestCase):
    def test_import(self):
        fname = 'test/data/0-0-ait-d0.990-h50.policy'
        #fname = 'test/data/0-0-appl-d0.990-tl240.policy'
        p = pomdp.POMDPPolicy(fname, file_format='aitoolbox', n_states=13)
        #p = pomdp.POMDPPolicy(fname, file_format='policyx', n_states=13)
        beliefs = [[0, 0.2, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]


        for b in beliefs:
            print b
            print p.get_action_rewards(b)
            #print p.get_best_action(b)

    def test_zmdp_import(self):
        fname = 'test/data/zmdp-term3.policy'
        p = pomdp.POMDPPolicy(fname, file_format='zmdp', n_states=4)
        print
        print '----------'
        print p.get_action_rewards([0.2, 0.2, 0.4, 0.2])
        print p.get_action_rewards([0.8, 0.0, 0.0, 0.2])
        print p.get_action_rewards([0.0, 0.8, 0.2, 0.0])



if __name__ == '__main__':
    unittest.main()
