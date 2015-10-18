import util

PEAKEDNESS = 1000

WEAK_PRIOR_MAG = 3.5
WEAK_BETA_MAG = WEAK_PRIOR_MAG * 2

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


#----------- HyperParams --------------
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


#----------- HyperParamsSpaced --------------
class HyperParamsSpaced(HyperParams):
    """Mostly uninformed priors, but worker accuracy spaced on [0, 0.5]."""
    def __init__(self, params, n_worker_classes, param_types_known=[]):
        super(HyperParamsSpaced, self).__init__(
            params, n_worker_classes, param_types_known)

        for k in self.p:
            t = get_param_type(k)
            if t == 'p_slip' and t not in param_types_known:
                if k[1] is None:
                    self.p[k] = list(util.beta_fit(
                        mode=0.25, mag=WEAK_BETA_MAG))
                else:
                    # Prior modes evenly spaced on [0, 0.5]
                    c = k[1]
                    self.p[k] = list(util.beta_fit(
                        mode=0.5*(c+1)/(n_worker_classes+1),
                        mag=WEAK_BETA_MAG))

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


#----------- HyperParamsSpacedStronger --------------
class HyperParamsSpacedStronger(HyperParamsSpaced):
    """Stronger prior on worker class probabilities."""
    def __init__(self, params, n_worker_classes, param_types_known=[]):
        super(HyperParamsSpacedStronger, self).__init__(
            params, n_worker_classes, param_types_known)

        for k in self.p:
            t = get_param_type(k)
            if t == 'p_worker':
                self.p[k] = [WEAK_PRIOR_MAG for i in xrange(n_worker_classes)]

class HyperParamsSpacedStrongerUnknownRatio(HyperParamsSpacedStronger):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedStrongerUnknownRatio, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_leave',
                                       'p_s'])

class HyperParamsSpacedStrongerUnknownRatioSlipLeave(HyperParamsSpacedStronger):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedStrongerUnknownRatioSlipLeave, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_s'])

class HyperParamsSpacedStrongerUnknownRatioSlipLeaveLose(HyperParamsSpacedStronger):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedStrongerUnknownRatioSlipLeaveLose, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_s'])
