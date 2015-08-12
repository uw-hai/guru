from flask import Flask, request, render_template
import pandas as pd
import os
import pickle
import networkx as nx
import json
from analyze import finished

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

def load_tree(f_exp, f_names, policy, episode=0):
    """Read an experiment output file and load a tree of execution traces"""
    df = finished(pd.read_csv(f_exp))
    df = df.dropna(subset=['a', 'o'])
    df[['a', 'o']] = df[['a', 'o']].astype(int)
    df = df[(df.policy == policy) & (df.episode == episode)]

    df_names = pd.read_csv(f_names,
                           true_values=['True', 'true'],
                           false_values=['False', 'false'])
    df_names_d = dict()
    for _, row in df_names.iterrows():
        df_names_d[row.type, row.i] = row.s

    G = nx.DiGraph()
    G.add_node(-1)  # Root, guaranteed to come before tuples when sorting.
    for i, df_i in df.groupby('iteration'):
        df_i = df_i.sort('t')
        seq = []
        for _, row in df_i.iterrows():
            a = ('action', row['a'])
            seq.append(a)
            if G.has_node(tuple(seq)):
                new_count = G.node[tuple(seq)]['count'] + 1
            else:
                new_count = 1
            G.add_node(tuple(seq), count=new_count,
                       label=df_names_d[seq[-1]],
                       type_=seq[-1][0])

            if len(seq) == 1:
                G.add_edge(-1, tuple(seq))
            else:
                G.add_edge(tuple(seq[:-1]), tuple(seq))

            o = ('observation', row['o'])
            seq.append(o)
            if G.has_node(tuple(seq)):
                new_count = G.node[tuple(seq)]['count'] + 1
            else:
                new_count = 1
            G.add_node(tuple(seq), count=new_count,
                       label=df_names_d[seq[-1]],
                       type_=seq[-1][0])
            G.add_edge(tuple(seq[:-1]), tuple(seq))
    return G

@app.route("/")
def index():
    exp = request.args.get('e')
    basename = request.args.get('f')
    policy = request.args.get('p')
    exp_filepath = os.path.join('res', exp, basename + '.txt')
    names_filepath = os.path.join('res', exp, basename + '_names.csv')
    tree = load_tree(f_exp=exp_filepath, f_names=names_filepath, policy=policy)
    tree = nx.convert_node_labels_to_integers(
        tree, first_label=-1, ordering='sorted')

    type_color = dict(observation='black', action='red')
    edges = []
    for n1, n2, _ in nx.to_edgelist(tree):
        edges.append([n2,
                      tree.node[n2]['label'],
                      n1,
                      tree.node[n2]['count'],
                      type_color[tree.node[n2]['type_']]])
    
    return render_template('test.html', exp=exp, basename=basename, policy=policy, edges=edges)
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
