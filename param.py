import copy
from . import util

PEAKEDNESS = 1000

WEAK_PRIOR_MAG = 3.5
WEAK_BETA_MAG = WEAK_PRIOR_MAG * 2

HYPERPARAMS = ['HyperParams',
               'HyperParamsUnknownRatio',
               'HyperParamsUnknownRatioLeave',
               'HyperParamsUnknownRatioSlipLeave',
               'HyperParamsUnknownRatioSlipLeaveLose',
               'HyperParamsUnknownRatioSlipLeaveLoseLearn',
               'HyperParamsUnknownRatioLeaveLose',

               'HyperParamsSpaced',
               'HyperParamsSpacedUnknownRatio',
               'HyperParamsSpacedUnknownRatioSlipLeave',
               'HyperParamsSpacedUnknownRatioSlipLeaveLose',
               'HyperParamsSpacedUnknownRatioLeaveLose',

               'HyperParamsSpacedStronger',
               'HyperParamsSpacedStrongerUnknownRatio',
               'HyperParamsSpacedStrongerUnknownRatioSlipLeave',
               'HyperParamsSpacedStrongerUnknownRatioSlipLeaveLose',
               'HyperParamsSpacedStrongerUnknownRatioLeaveLose']

class Params(object):
    """Class for configuration params."""
    def __init__(self, config):
        self.params = config
        self.n_classes = len(config['p_worker'])

    def get_param_dict(self, sample=False):
        """Return dictionary without p_slip_std.

        Args:
            sample: Sample p_slip from truncated normal standard deviation
                    specified by p_slip_std. (Otherwise, use mean p_slip.)

        """
        res = dict()
        for k in self.params:
            if self.get_param_type(k) == 'p_slip' and sample:
                try:
                    std = self.params['p_slip_std', None]
                except KeyError:
                    std = self.params['p_slip_std', k[1]]
                p = util.truncnorm_sample(
                    lower=0, upper=0.5,
                    mu=self.params[k][0], std=std, size=1)[0]
                res[k] = [p, 1-p]
            elif self.get_param_type(k) != 'p_slip_std':
                res[k] = copy.copy(self.params[k])
        return res

    @classmethod
    def from_cmd(cls, config):
        """Convert command line config parameters to params for POMDPModel.

        Notes:
        - 'p_worker' must give full categorical probability vector.
        - Other probabilities are bernoulli distributions and must be given
          only using positive probability.
        - Bernoulli distributions can either be conditioned on p_worker, or
          not.
        - Defaults 'reward_tp' = 'reward_'tn' = 1.

        Infers whether Bernoulli distributions are conditioned or use parameter
        tying from the number of parameters specified.

        Args:
            config: Dictionary of command line config parameters.

        Returns:
            New dictionary of parameters.
        """
        # Add reward for legacy reasons.
        if 'penalty_fp' in config:
            for k in ['reward_tp', 'reward_tn']:
                if k not in config:
                    config[k] = 1

        n_worker_classes = len(config['p_worker'])
        n_rules = len(config['p_r'])

        # Copy dictionary and split p_s by rule.
        res = dict()
        for k in config:
            if k == 'p_s':
                if (len(config[k]) != n_rules and
                    len(config[k]) != n_rules * n_worker_classes):
                    raise Exception('Config input of unexpected size')
                for i, v in enumerate(config[k]):
                    if i < n_rules:
                        res[k, i] = []
                    res[k, i % n_rules].append(v)
            else:
                # All other parameters with list values must either be
                # length 1 or |classes|.
                if (isinstance(config[k], list) and len(config[k]) > 1 and
                    len(config[k]) != n_worker_classes):
                    raise Exception('Config input {} of unexpected size'.format(k))
                res[k] = config[k]

        # Split out classes.
        # Make berunoulli probabilities full probabilities.
        for k in res.keys():
            if (k in ['p_learn_exp', 'p_learn_tell', 'p_lose',
                      'p_leave', 'p_slip', 'p_guess', 'p_slip_std'] or
                (len(k) == 2 and k[0] == 'p_s')):
                values = res.pop(k)
                if len(values) == 1:
                    if k == 'p_slip_std':
                        res[k, None] = values[0]
                    else:
                        res[k, None] = [values[0], 1 - values[0]]
                else:
                    for i, v in enumerate(values):
                        if k == 'p_slip_std':
                            res[k, i] = v
                        else:
                            res[k, i] = [v, 1 - v]

        return cls(res)

    @staticmethod
    def get_param_type(p):
        """Get type of param.

        >>> Params.get_param_type('p_worker')
        'p_worker'
        >>> Params.get_param_type(('p_guess', None))
        'p_guess'
        >>> Params.get_param_type((('p_s', 2), 3))
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
            t = Params.get_param_type(k)
            if t in param_types_known:
                # Make peaked dirichlet at parameters.
                p[k] = [1.00001 + PEAKEDNESS * v for v in params[k]]
            elif t == 'p_worker':
                p[k] = [1.00001 for i in xrange(n_worker_classes)]
            #elif t == 'p_guess':
            #    p[k] = [10, 10] # Pretty sure this is 0.5.
            elif t == 'p_slip':
                p[k] = [2, 5] # Lower prob of making a mistake.
            elif t == 'p_lose':
                p[k] = [2, 20] # Lower prob of losing a skill.
            elif t in ['p_learn_exp', 'p_learn_tell', 'p_leave',
                       'p_s', 'p_guess']:
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

class HyperParamsUnknownRatioLeave(HyperParams):
    """Hyperparameters with unknown class ratio and p_leave."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsUnknownRatioLeave, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
                                       'p_lose',
                                       'p_learn_exp',
                                       'p_learn_tell',
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

class HyperParamsUnknownRatioSlipLeaveLoseLearn(HyperParams):
    def __init__(self, params, n_worker_classes):
        super(HyperParamsUnknownRatioSlipLeaveLoseLearn, self).__init__(
            params, n_worker_classes, ['p_learn_tell'])

class HyperParamsUnknownRatioLeaveLose(HyperParams):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsUnknownRatioLeaveLose, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
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
            t = Params.get_param_type(k)
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

class HyperParamsSpacedUnknownRatioLeaveLose(HyperParamsSpaced):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedUnknownRatioLeaveLose, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
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
            t = Params.get_param_type(k)
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

class HyperParamsSpacedStrongerUnknownRatioLeaveLose(HyperParamsSpacedStronger):
    """Hyperparameters with known class properties but unknown ratio."""
    def __init__(self, params, n_worker_classes):
        super(HyperParamsSpacedStrongerUnknownRatioLeaveLose, self).__init__(
            params, n_worker_classes, ['p_guess',
                                       'p_slip',
                                       'p_learn_exp',
                                       'p_learn_tell',
                                       'p_s'])
