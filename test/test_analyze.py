import unittest
import os
import pandas as pd
import analyze

data_base = os.path.join('test', 'data', 'sample_res', 'zmdp-discount_0.99-timeout_60-teach_first-teach_type_tell-n_0-test_and_boot-n_work_20-accuracy_0.8-n_test_4')

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


if __name__ == '__main__':
    unittest.main()
