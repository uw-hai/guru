from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import os
import pickle
import networkx as nx
import numpy as np
import json
from analyze import finished

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

def belief_to_str(states, belief):
    return str(dict((s, b) for s, b in zip(states, belief)))

def load_tree(f_exp, f_names, policy, episode=0):
    """Read an experiment output file and load a tree of execution traces"""
    df = finished(pd.read_csv(f_exp))
    df = df[(df.policy == policy) & (df.episode == episode)]

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
    exp = request.args.get('e')
    basename = request.args.get('f')
    policy = request.args.get('p')
    episode = request.args.get('episode', default=0)
    exp_filepath = os.path.join('res', exp, basename + '.txt')
    names_filepath = os.path.join('res', exp, basename + '_names.csv')

    # Wordtree visualization.
    tree = load_tree(f_exp=exp_filepath,
                     f_names=names_filepath,
                     policy=policy,
                     episode=episode)
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
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
