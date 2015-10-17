# -*- coding: utf-8 -*-
"""
    pypomdp.util.zmdp
    ~~~~~~~~~~~~~~~~~

    :copyright: (c) 2012 by Bastian Migge, Santiago Droll
    :license: BSD3, see LICENSE for more details.
    :description: zMDP utilities (http://www.cs.cmu.edu/~trey/zmdp/)
"""

import re
import yaml
import io

def read_zmdp_policy(filename, state_count):
    """
    return list of alphavectors and corresponding actions from zMDP policy file

    zMDP info: http://www.cs.cmu.edu/~trey/zmdp/

    arguments:
        filename
    """
    alpha_vectors, alpha_vector_actions = [], []
    yaml_data = io.StringIO()
    yaml_str = ""
    # read and convert zMDP policy format to YAML
    with open(filename,'r') as f:
        zmdp_data = f.read()
        yaml_data.write( u'%s' % re.sub('=>',':',zmdp_data) )
        yaml_str += re.sub('=>',':',zmdp_data)

    # read YAML
    dataMap = yaml.load(yaml_str)

    # sanity check
    assert dataMap['policyType'] == 'MaxPlanesLowerBound', 'unrecognized policy'

    # generate cassandra alpha string
    for plane in dataMap['planes']:
        alpha_vector_actions.append(int(plane['action']))
        # since the mdp value is defined only if non zero, we must
        # manually define the zero values of the alpha vector
        alpha_vector = [None] * state_count
        for index in range(len(plane['entries'])/2):
            actionId = int(plane['entries'][index*2])
            mdp_value = float(plane['entries'][index*2+1])
            alpha_vector[actionId] = mdp_value
        alpha_vectors.append(alpha_vector)

    yaml_data.close()
    return alpha_vector_actions, alpha_vectors
