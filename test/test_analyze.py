import unittest
import os
import pandas as pd
import pymongo
from .. import analyze
from .. import util

TEST_EXPERIMENT = 'classes2-20_80'

# TODO: Make test data that does not depend on private database. Tests
# below commented until this happens.
"""
class AuthenticatedTestCase(unittest.TestCase):
    mongo = {'host': os.environ['MONGO_HOST'],
             'port': int(os.environ['MONGO_PORT']),
             'user': os.environ.get('MONGO_USER', None),
             'pass': os.environ.get('MONGO_PASS', None)}
    mongo['dbname'] = os.environ.get('MONGO_DBNAME')
    mongo['auth_dbname'] = os.environ.get('MONGO_AUTH_DBNAME',
                                          mongo['dbname'])
    client = pymongo.MongoClient(mongo['host'], mongo['port'])
    if mongo['user']:
        client[mongo['auth_dbname']].authenticate(mongo['user'], mongo['pass'],
                                                  mechanism='SCRAM-SHA-1')

class PlotterTestCase(AuthenticatedTestCase):
    def setUp(self):
        self.plotter = analyze.Plotter.from_mongo(
            collection=self.client[self.mongo['dbname']].res,
            experiment=TEST_EXPERIMENT,
            collection_names=self.client[self.mongo['dbname']].names)

    def test_filter_workers_quantile(self):
        orig_len = len(self.plotter.df)
        small = self.plotter.filter_workers_quantile(self.plotter.df, 0.75, 1)
        med = self.plotter.filter_workers_quantile(self.plotter.df, 0.6, 1)
        full = self.plotter.filter_workers_quantile(self.plotter.df, 0, 1)
        self.assertEqual(orig_len, len(full))
        self.assertGreater(len(full), len(med))
        self.assertGreater(len(med), len(small))
        self.assertLess(0, len(small))

class ResultPlotterTestCase(AuthenticatedTestCase):
    def setUp(self):
        self.plotter = analyze.ResultPlotter.from_mongo(
            collection=self.client[self.mongo['dbname']].res,
            experiment=TEST_EXPERIMENT,
            collection_names=self.client[self.mongo['dbname']].names)
        self.path = os.path.join('test', 'tmp', 'plots', 'res')
        util.ensure_dir(self.path)


    def test_plot_reward_by_budget(self):
        self.plotter.plot_reward_by_budget(
            os.path.join(self.path, 'r_cost'), fill=True)

    def test_make_plots(self):
        self.plotter.make_plots(self.path, line=False, logx=True,
                                worker_interval=5)

class ModelPlotterTestCase(AuthenticatedTestCase):
    def setUp(self):
        self.plotter = analyze.ModelPlotter.from_mongo(
            collection=self.client[self.mongo['dbname']].model,
            experiment=TEST_EXPERIMENT)

    def test_load(self):
        self.assertLess(0, len(self.plotter.df))

    def test_make_plots(self):
        path = os.path.join('test', 'tmp', 'plots', 'model')
        util.ensure_dir(path)
        self.plotter.make_plots(path)

#class JoinPlotterTestCase(AuthenticatedTestCase):
#    def setUp(self):
#        df = pd.read_csv('{}.txt'.format(data_base_rl))
#        df_names = pd.read_csv('{}_names.csv'.format(data_base_rl))
#        df_model = pd.read_csv('{}_model.csv'.format(data_base_rl))
#        df_timings = pd.read_csv('{}_timings.csv'.format(data_base_rl))
#        self.plotter = analyze.JoinPlotter(df, df_model, df_timings, df_names)
#
#    def test_get_traces(self):
#        p = self.plotter.df.policy.unique()[0]
#        itr = self.plotter.get_traces(p)
#        itr.next()

class TimingsPlotterTestCase(AuthenticatedTestCase):
    def setUp(self):
        self.collection = self.client[self.mongo['dbname']].timing
        self.plotter = analyze.TimingsPlotter.from_mongo(
             self.collection, experiment=TEST_EXPERIMENT)

    def test_load(self):
        self.assertLess(0, len(self.plotter.df))

    def test_make_plots(self):
        path = os.path.join('test', 'tmp', 'plots', 'timings')
        util.ensure_dir(path)
        self.plotter.make_plots(os.path.join(path, 't'))
"""

if __name__ == '__main__':
    unittest.main()
