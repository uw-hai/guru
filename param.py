import util

PEAKEDNESS = 1000

WEAK_BETA_MAG = 7

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
    """Mostly uninformed priors."""
    def __init__(self, params, n_worker_classes, param_types_known=[]):
        p = dict()
        for k in params:
            t = get_param_type(k)
            if t in param_types_known:
                # Make peaked dirichlet at parameters.
                p[k] = [1.00001 + PEAKEDNESS * v for v in params[k]]
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

class HyperParamsSpaced(object):
    """Mostly uninformed priors, but worker accuracy spaced on [0, 0.5]."""
    def __init__(self, params, n_worker_classes, param_types_known=[]):
        p = dict()
        for k in params:
            t = get_param_type(k)
            if t in param_types_known:
                # Make peaked dirichlet at parameters.
                p[k] = [1.00001 + PEAKEDNESS * v for v in params[k]]
            elif t == 'p_worker':
                p[k] = [1.00001 for i in xrange(n_worker_classes)]
            elif t == 'p_guess':
                p[k] = [10, 10] # Pretty sure this is 0.5.
            elif t == 'p_slip':
                if k[1] is None:
                    p[k] = list(util.beta_fit(mode=0.25, mag=WEAK_BETA_MAG))
                else:
                    # Prior modes evenly spaced on [0, 0.5]
                    c = k[1]
                    p[k] = list(util.beta_fit(
                        mode=0.5*(c+1)/(n_worker_classes+1),
                        mag=WEAK_BETA_MAG))
            elif t in ['p_lose', 'p_learn_exp', 'p_learn_tell', 'p_leave',
                       'p_s']:
                p[k] = [1.00001, 1.00001]
        self.p = p

class HyperParamsWorker5(object):
    """Mostly uninformed priors, except Dirichlet(5) for p_worker."""
    def __init__(self, params, n_worker_classes, param_types_known=[]):
        p = dict()
        for k in params:
            t = get_param_type(k)
            if t in param_types_known:
                # Make peaked dirichlet at parameters.
                p[k] = [1.00001 + PEAKEDNESS * v for v in params[k]]
            elif t == 'p_worker':
                p[k] = [5 for i in xrange(n_worker_classes)]
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

class HyperParamsUnknownRatioSlipLeave(HyperParams):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsUnknownRatioSlipLeave, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_s'])

class HyperParamsUnknownRatioSlipLeaveLose(HyperParams):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsUnknownRatioSlipLeaveLose, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_s'])

class HyperParamsSpacedUnknownRatio(HyperParamsSpaced):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedUnknownRatio, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_leave',
                                       'p_s'])

class HyperParamsSpacedUnknownRatioSlipLeave(HyperParamsSpaced):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedUnknownRatioSlipLeave, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_s'])

class HyperParamsSpacedUnknownRatioSlipLeaveLose(HyperParamsSpaced):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedUnknownRatioSlipLeaveLose, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_s'])


class HyperParamsUnknownRatioWorker5(HyperParamsWorker5):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsUnknownRatioWorker5, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_leave',
                                       'p_s'])
