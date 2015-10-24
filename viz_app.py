import collections
import flask
from flask import Flask, request, render_template, send_from_directory
from flask.ext.pymongo import PyMongo
import pandas as pd
import os
import pickle
import networkx as nx
import numpy as np
import json
import util
import analyze

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)
app.config.from_object(os.environ['APP_SETTINGS'])
mongo = PyMongo(app)


PLOTDIR = os.path.join(app.config['STATIC_FOLDER'], 'plots')

def belief_to_str(states, belief, threshold=0):
    """Return belief string.

    >>> belief_to_str(['a', 'b'], ['0.3', '0.7'])
    'b (0.700), a (0.300)'
    >>> belief_to_str(['a', 'b'], ['0.7', '0.3'])
    'a (0.700), b (0.300)'
    >>> belief_to_str(['a', 'b'], ['0.7', '0.3'], threshold=0.4)
    'a (0.700)'
    >>> belief_to_str(['a', 'b'], ['0.3', '0.7'], threshold=0.4)
    'b (0.700)'

    """
    state_to_belief = dict((
        (s, b) for (s, b) in zip(states, (float(b) for b in belief)) if
        float(b) > threshold))
    return ', '.join('{} ({:0.3f})'.format(k, state_to_belief[k]) for k in
           sorted(state_to_belief, key=lambda x: state_to_belief[x],
                  reverse=True))

def load_tree(f_exp, f_names, policy):
    """Read an experiment output file and load a tree of execution traces"""
    df = pd.read_csv(f_exp)
    df = df[df.policy == policy]

    df_names = pd.read_csv(f_names,
                           true_values=['True', 'true'],
                           false_values=['False', 'false'])
    df_names_d = dict()
    for _, row in df_names.iterrows():
        df_names_d[row.type, row.i] = row.s
    state_names = [df_names_d[x] for x in sorted(df_names_d) if
                   x[0] == 'state']

    G = nx.DiGraph()
    for i, df_i in df.groupby('iteration'):
        df_i = df_i.sort('t')
        current_belief = belief_to_str(state_names, df_i['b'].iloc[0].split())
        cum_r = 0
        # -1 state guaranteed before tuples.
        if not G.has_node(-1):
            G.add_node(-1)
        df_i = df_i.dropna(subset=['a', 'o'])
        df_i[['a', 'o']] = df_i[['a', 'o']].astype(int)

        seq = []
        for _, row in df_i.iterrows():
            a = ('action', row['a'])
            seq.append(a)
            if G.has_node(tuple(seq)):
                new_count = G.node[tuple(seq)]['count'] + 1
                new_cum_reward = G.node[tuple(seq)]['cum_reward'] + [cum_r]
            else:
                new_count = 1
                new_cum_reward = [cum_r]
            G.add_node(tuple(seq),
                       count=new_count,
                       t=row['t'],
                       label=df_names_d[seq[-1]],
                       type_=seq[-1][0],
                       belief=current_belief,
                       cum_reward=new_cum_reward,
                       reward=[0])

            if len(seq) == 1:
                G.add_edge(-1, tuple(seq))
            else:
                G.add_edge(tuple(seq[:-1]), tuple(seq))

            o = ('observation', row['o'])
            seq.append(o)
            current_belief = belief_to_str(state_names, row['b'].split())
            cum_r += float(row['r'])
            if G.has_node(tuple(seq)):
                new_count = G.node[tuple(seq)]['count'] + 1
                new_cum_reward = G.node[tuple(seq)]['cum_reward'] + [cum_r]
                new_reward = G.node[tuple(seq)]['reward'] + [float(row['r'])]
            else:
                new_count = 1
                new_cum_reward = [cum_r]
                new_reward = [float(row['r'])]
            G.add_node(tuple(seq),
                       count=new_count,
                       t=row['t'],
                       label=df_names_d[seq[-1]],
                       type_=seq[-1][0],
                       belief=current_belief,
                       cum_reward=new_cum_reward,
                       reward=new_reward)
            G.add_edge(tuple(seq[:-1]), tuple(seq))
    return G

@app.route("/")
def index():
    """Return list of experiment"""
    return '\n'.join(mongo.db.res.distinct('experiment'))

@app.route("/viz")
def viz():
    exp = request.args.get('e')
    basename = request.args.get('f')
    policy = request.args.get('p')
    exp_filepath = os.path.join('res', exp, basename + '.txt')
    names_filepath = os.path.join('res', exp, basename + '_names.csv')

    # Wordtree visualization.
    tree = load_tree(f_exp=exp_filepath,
                     f_names=names_filepath,
                     policy=policy)
    print 'done loading tree'
    tree = nx.convert_node_labels_to_integers(tree,
                                              first_label=-1,
                                              ordering='sorted')
    type_color = dict(observation='black', action='red')
    edges = []
    for n1, n2, _ in nx.to_edgelist(tree):
        edges.append(
            [n2,
             tree.node[n2]['label'],
             n1,
             tree.node[n2]['count'],
             type_color[tree.node[n2]['type_']]])
    print 'done loading wordtree edgelist'

    # Treemap visualization.
    edges_treemap = []
    edges_treemap_other = []
    for n1, n2, _ in nx.to_edgelist(tree):
        edges_treemap.append(
            [{'v': str(n2), 'f': tree.node[n2]['label']},
             None if n1 == -1 else str(n1),
             tree.node[n2]['count'],
             np.mean(tree.node[n2]['cum_reward'])])
        edges_treemap_other.append(
            [tree.node[n2]['belief'],
             tree.node[n2]['t'],
             np.mean(tree.node[n2]['reward'])])
    print 'done loading treemap edgelist'
    # Other plots.
    #png_actions = os.path.join(
    #    exp, basename, 'a_e{}_p-{}.png'.format(episode, policy))
    #png_rewards = os.path.join(
    #    exp, basename, 'r_t_e{}.png'.format(episode))

    return render_template(
        'test.html', exp=exp, basename=basename,
        policy=policy, edges=edges, edges_treemap=edges_treemap,
        edges_treemap_other=edges_treemap_other)

@app.route('/status')
def status():
    # BUG: Slow.
    e = request.args.get('e')
    if e:
        experiments = [e]
    else:
        experiments = mongo.db.res.distinct('experiment')
    d = dict()
    for e in experiments:
        d[e] = dict()
        policies = mongo.db.res.find({'experiment': e}).distinct('policy')
        for p in policies:
            iters = mongo.db.res.find(
                {'experiment': e, 'policy': p}).distinct('iteration')
            d[e][p] = len(iters)
    return render_template('status.html', iterations=d)

@app.route('/iterations')
def iterations():
    e = request.args.get('e')
    p = request.args.get('p')
    return ' '.join(str(x) for x in mongo.db.res.find(
        {'experiment': e, 'policy': p}).distinct('iteration'))


@app.route('/plots')
def plots():
    e = request.args.get('e')
    if e:
        exps = [e]
    else:
        exps = os.listdir(PLOTDIR)
    d = dict()
    for e in exps:
        expdir = os.path.join(PLOTDIR, e)
        plots = [f for f in os.listdir(expdir) if f.endswith('.png')]
        d[e] = plots
    return render_template('plots.html', files=d)

@app.route('/make_plots')
def make_plots():
    e = request.args.get('e')
    if e:
        experiments = [e]
    else:
        experiments = mongo.db.res.distinct('experiment')
    for e in experiments:
        analyze.make_plots(
            db=mongo.db, experiment=e, outdir=os.path.join(PLOTDIR, e))

@app.route('/traces')
def traces():
    e = request.args.get('e')
    i = request.args.get('i')
    p = request.args.get('p')
    if not e:
        return 'choose an experiment e'
    if not i:
        return 'choose an iteration i'
    if not p:
        return 'choose a policy p'
    res = list(mongo.db.res.find(
        {'experiment': e, 'policy': p, 'iteration': int(i)},
        {'_id': False}).sort(
            't', flask.ext.pymongo.ASCENDING))
    model = list(mongo.db.model.find(
        {'experiment': e, 'policy': p, 'iteration': int(i)},
        {'_id': False}).sort('worker', flask.ext.pymongo.ASCENDING))
    model_by_worker = collections.defaultdict(dict)
    for r in model:
        model_by_worker[r['worker']][r['param']] = r['v']
    for w in model_by_worker:
        model_by_worker[w] = ['{}: {}'.format(k, v) for
            k, v in sorted(model_by_worker[w].iteritems())]
    st_dict = dict((x['i'], x['s']) for x in mongo.db.names.find(
        {'experiment': e, 'type': 'state'}, {'_id': False}))
    st = [st_dict[x] for x in sorted(st_dict)]
    obs_dict = dict(list((x['i'], x['s']) for x in mongo.db.names.find(
        {'experiment': e, 'type': 'observation'}, {'_id': False})))
    action_dict = dict(list((x['i'], x['s']) for x in mongo.db.names.find(
        {'experiment': e, 'type': 'action'}, {'_id': False})))
    res_by_worker = collections.defaultdict(list)
    for r in res:
        res_by_worker[r['worker']].append(r)
    for w, res in res_by_worker.iteritems():
        beliefs = [belief_to_str(st, x['b'], threshold=0.01) for x in res]
        observations = [
            '' if x['o'] is None else obs_dict[x['o']] for x in res]
        actions = [
            '' if x['a'] is None else action_dict[x['a']] for x in res]
        rewards = [
            '' if x['r'] is None else x['r'] for x in res]
        costs = [
            '' if x['cost'] is None else x['cost'] for x in res]
        explore = [
            '' if x['explore'] is None else x['explore'] for x in res]
        reserved = [
            '' if 'reserved' not in x or x['reserved'] is None else
            x['reserved'] for x in res]
        res_by_worker[w] = zip(
            actions, observations, beliefs, rewards, costs, explore, reserved)
    return render_template(
        'traces.html',
        res=res_by_worker,
        model=model_by_worker)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
