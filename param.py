PEAKEDNESS = 1000

def get_param_type(p):
    """Get type of param.

    >>> get_param_type('p_worker')
    'p_worker'
    >>> get_param_type(('p_guess', None))
    'p_guess'
    >>> get_param_type((('p_s', 2), 3))
    'p_s'

    """
    if not isinstance(p, tuple):
        return p
    elif not isinstance(p[0], tuple):
        return p[0]
    else:
        return p[0][0]

class HyperParams(object):
    def __init__(self, params, n_worker_classes, param_types_known=[]):
        p = dict()
        for k in params:
            t = get_param_type(k)
            if t in param_types_known:
                # Make peaked dirichlet at parameters.
                p[k] = [1 + PEAKEDNESS * v for v in params[k]]
            elif t == 'p_worker':
                p[k] = [1.00001 for i in xrange(n_worker_classes)]
            elif t == 'p_guess':
                p[k] = [10, 10] # Pretty sure this is 0.5.
            elif t == 'p_slip':
                p[k] = [2, 5] # Lower prob of making a mistake.
            elif t in ['p_lose', 'p_learn_exp', 'p_learn_tell', 'p_leave',
                       'p_s']:
                p[k] = [1.00001, 1.00001]
        self.p = p

class HyperParamsUnknownRatio(HyperParams):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsUnknownRatio, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_leave',
                                       'p_s'])
