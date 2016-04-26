import os
import unittest
import json
import copy
import numpy.testing as npt
from .. import param

class TestParams(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), os.pardir, 'config', 'classes2-20_80_cost_0.000001.json'), 'r') as f:
            self.d = json.load(f)

        self.d2 = copy.deepcopy(self.d)
        self.d2['p_slip_std'] = [0, 0.05]

    def test_no_p_slip_std(self):
        params = param.Params.from_cmd(self.d)
        means = params.get_param_dict(sample=False)

        self.assertDictEqual(params.params, means)
        self.assertNotIn(means, ('p_slip_std', None))
        self.assertNotIn(means, ('p_slip_std', 0))

    def test_p_slip_std(self):
        params = param.Params.from_cmd(self.d2)

        means = params.get_param_dict(sample=False)
        print means
        npt.assert_almost_equal(means[('p_slip', 0), 0], params.params[('p_slip', 0), 0])
        npt.assert_almost_equal(means[('p_slip', 0), 1], params.params[('p_slip', 0), 1])

        sampled = params.get_param_dict(sample=True)
        npt.assert_almost_equal(sampled[('p_slip', 0), 0],
                                params.params[('p_slip', 0), 0])
        self.assertNotAlmostEqual(sampled[('p_slip', 0), 1][0],
                                  params.params[('p_slip', 0), 1][0])
