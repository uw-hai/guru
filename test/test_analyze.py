import unittest
import os
import pandas as pd
import analyze

data_base = os.path.join('test', 'data', 'sample_res', 'zmdp-discount_0.99-timeout_60-teach_first-teach_type_tell-n_0-test_and_boot-n_work_20-accuracy_0.8-n_test_4')
data_base_rl = os.path.join('test', 'data', 'sample_res', 'zmdp-discount_0.99-timeout_60-eps_1div(0.1*w+1)-HyperParamsUnknownRatio')

class PlotterTestCase(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('{}.txt'.format(data_base))
        df_names = pd.read_csv('{}_names.csv'.format(data_base))
        self.plotter = analyze.Plotter(df, df_names)

    def test_filter_workers_quantile(self):
        orig_len = len(self.plotter.df)
        small = self.plotter.filter_workers_quantile(self.plotter.df, 0.75, 1)
        med = self.plotter.filter_workers_quantile(self.plotter.df, 0.6, 1)
        full = self.plotter.filter_workers_quantile(self.plotter.df, 0, 1)
        self.assertEqual(orig_len, len(full))
        self.assertGreater(len(full), len(med))
        self.assertGreater(len(med), len(small))
        self.assertLess(0, len(small))

class ModelPlotterTestCase(unittest.TestCase):
    def setUp(self):
        df_model = pd.read_csv('{}_model.csv'.format(data_base_rl))
        self.plotter = analyze.ModelPlotter(df_model)

class JoinPlotterTestCase(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('{}.txt'.format(data_base_rl))
        df_names = pd.read_csv('{}_names.csv'.format(data_base_rl))
        df_model = pd.read_csv('{}_model.csv'.format(data_base_rl))
        df_timings = pd.read_csv('{}_timings.csv'.format(data_base_rl))
        self.plotter = analyze.JoinPlotter(df, df_model, df_timings, df_names)

    def test_get_traces(self):
        p = self.plotter.df.policy.unique()[0]
        itr = self.plotter.get_traces(p)
        itr.next()


if __name__ == '__main__':
    unittest.main()
