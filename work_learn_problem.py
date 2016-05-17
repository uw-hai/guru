"""Helper functions for specifying a work-learn problem.

Port from WorkLearnProblem.hpp

"""

from __future__ import division
import collections
import itertools
import numpy as np

NINF = -999
# TODO: Change quiz_val to rule / skill.


class Action:

    def __init__(self, name, quiz_val=None):
        self.name = name
        self.quiz_val = quiz_val

    def get_type(self):
        """Return the action type"""
        if self.name == 'ask' and self.quiz_val is None:
            return 'work'
        elif self.is_quiz():
            return 'test'
        else:
            return self.name

    def is_quiz(self):
        return self.name == 'ask' and self.quiz_val is not None

    def uses_gold(self):
        """Return whether an action uses a gold question.

        Does not count explain actions as using gold, since they use the
        same gold question as the preceding ask.

        """
        return self.name == 'tell' or (self.name == 'ask' and
                                       self.quiz_val is not None)

    def valid_after(self, action=None):
        """Return whether this action may follow the provided action.

        If action is None, return whether the action may be the first action.

        """
        return self.name != 'exp' or (action is not None and action.is_quiz())

    def __str__(self):
        s = self.name
        if self.quiz_val is not None:
            s += '-rule_{}'.format(self.quiz_val)
        return s

    def __eq__(self, a):
        return self.name == a.name and self.quiz_val == a.quiz_val


def actions_all(n_skills, n_question_types=1, tell=False, exp=False):
    """Return all actions

    Args:
        n_skills (int): Number of skills.
        n_question_types (int): Number of question types. If more than one
            question type, create a single quiz / teach action.
        tell (bool): Include tell actions.
        exp (bool): Include explain action.

    Returns:
        [work_learn_problem.Action]: List of actions.

    """
    actions = [Action('boot'), Action('ask', None)]
    if n_question_types > 1:
        n_skills = 1
    actions += [Action('ask', i) for i in xrange(n_skills)]
    if exp:
        actions.append(Action('exp'))
    if tell:
        actions += [Action('tell', i) for i in xrange(n_skills)]
    return actions

# New observations ['yes', 'no'] for ask.
#observations = ['yes', 'no', 'wrong', 'right', 'term']
#observations = ['wrong', 'right', 'null', 'term']
def observations(n_question_types=1):
    """Return observation set.

    Args:
        n_question_types (int): Number of question types.

    Returns:
        observations ([str]): Observation strings. Strings are either 'null',
            'term', or a string like 'rwr' which indicates "right", "wrong',
            and "right" answers to question types 1, 2, and 3, respectively.

    """
    out = ['null', 'term']
    out += [''.join(x) for x in
            itertools.product(['r', 'w'], repeat=n_question_types)]
    return out


def states_all(n_skills, n_worker_classes, n_question_types=1):
    """Enumerate states.

    Args:
        n_skills (int): Number of skills.
        n_worker_classes (int): Number of worker classes.
        n_question_types (int): Number of question types. If more than one
            question type, create a single quiz value in the state.


    Returns:
        states ([work_learn_problem.State]): List of states.

    """
    skill_values = list(itertools.product((True, False), repeat=n_skills))
    if n_question_types > 1:
        quiz_values = [None, 0]  # Matches single action in actions_all().
    else:
        quiz_values = [None] + range(n_skills)
    worker_class_values = range(n_worker_classes)
    states_except_term = [
        State(skills=s, quiz_val=q, worker_class=w) for s, q, w in
        itertools.product(skill_values, quiz_values, worker_class_values)]
    return [State(term=True)] + states_except_term


class State:

    def __init__(self, term=False, skills=None, quiz_val=None,
                 worker_class=None):
        """Init.

        Args:
            term (bool): Is a terminal state.
            skills ([bool]): List indicating whether worker has each skill.
            quiz_val (Optional[int])): Quiz action last taken (or None).
            worker_class (int): Worker class.

        """
        if skills is None:
            skills = []
        self.term = term
        self.skills = skills
        self.quiz_val = quiz_val
        self.worker_class = worker_class

    def has_skill(self, skill):
        if self.term:
            raise Exception('Terminal state has no skill')
        return self.skills[skill]

    def is_quiz(self):
        if self.term:
            return False
        return self.quiz_val is not None

    def p_answer_correctly(self):
        raise NotImplementedError

    def is_valid_action(self, action):
        if self.term:
            return True
        elif action.name == 'exp' and not self.is_quiz():
            return False
        else:
            return True

    def n_skills(self):
        return len(self.skills)

    def n_skills_known(self):
        return sum(self.skills)

    def n_skills_learned(self, next_state):
        return len(self.skills_learned(next_state))

    def n_skills_lost(self, next_state):
        return len(self.skills_lost(next_state))

    def skills_learned(self, next_state):
        return [i for i, (x, y) in
                enumerate(zip(self.skills, next_state.skills)) if not x and y]

    def skills_lost(self, next_state):
        return [i for i, (x, y) in
                enumerate(zip(self.skills, next_state.skills)) if x and not y]

    def has_same_skills(self, next_state):
        return self.skills == next_state.skills

    def p_has_skills(self, rule_probabilities):
        """Probability of having necessary skills"""
        if self.term:
            raise Exception('Unexpected terminal state')
        p_has_skills = 1
        for i, p in enumerate(rule_probabilities):
            if not self.has_skill(i):
                p_has_skills *= 1 - p
        return p_has_skills

    def p_right(self, rule_probabilities, p_slip, p_guess):
        # TODO: Move to separate class?
        """Probability of answering correctly"""
        if self.term:
            raise Exception('Unexpected terminal state')
        p_has_skills = self.p_has_skills(rule_probabilities)
        return p_has_skills * (1 - p_slip) + (1 - p_has_skills) * p_guess

    def p_joint(self, rule_probabilities, p_slip, p_guess,
                prior, answer, observation):
        # TODO: Move to separate class?
        """Joint probability of latent answer and observation

        Args:
            answer: 0 or 1
            observation: 0 or 1

        """
        if self.term:
            raise Exception('Unexpected terminal state')
        if answer == 0:
            p = 1 - prior
        else:
            p = prior
        p_right = self.p_right(rule_probabilities, p_slip, p_guess)
        if observation == answer:
            p *= p_right
        else:
            p *= 1 - p_right
        return p

    def rewards_ask(self, p_r, p_slip, p_guess, priors, utility_type,
                    penalty_fp, penalty_fn, reward_tp, reward_tn, sample):
        """Return expected reward and sampled additional info.

        Args:
            p_r ([float]): Probability each rule is needed for a question.
            p_slip ([float]): Probability of answering incorrectly if a rule
                is known, one value for each question type.
            p_guess ([float]): Probability of guessing correctly, one value
                for each question type.
            priors ([float]): Prior probability true answer is "1" for each
                question type.
            utility_type (str): Utility type.
            penalty_fp (float): False positive penalty.
            penalty_fn (float): False negative penalty.
            reward_tp (float): True positive reward.
            reward_tn (float): True negative reward.
            sample (bool): Sample instead of expected reward.

        Returns:
            rewards ([float]): Reward (expected reward if sample is False)
                for each question type.
            metadata ([dict]): Metadata for each question type. Contains
                'rewards', 'gt', 'answer', with a list for each question
                type.

        """
        if utility_type == 'pen_nonboolean' and reward_tp != reward_tn or penalty_fp != penalty_fn:
            raise Exception(
                "Rewards differ for boolean outcomes, but using nonboolean reward function")

        metadata = collections.defaultdict(list)
        n_question_types = len(priors)
        n_rules = len(p_r)
        # If len(priors) > 1, then p_r is all 0 except for question type index.
        if n_question_types == 1:
            p_r_question_types = [p_r]
        elif n_rules == 1:
            p_r_question_types = [[p_r[0]]] * n_question_types
        else:
            p_r_question_types = p_r * np.eye(n_rules)
        p_slips = p_slip
        p_guesses = p_guess
        for prior, p_slip, p_guess, p_r in zip(
                priors, p_slips, p_guesses, p_r_question_types):
            if sample:  # Sample label values.
                v = []
                probs = []
                for a in (0, 1):
                    for o in (0, 1):
                        v.append({'gt': a, 'answer': o})
                        probs.append(self.p_joint(
                            p_r, p_slip, p_guess, prior, a, o))
                v_sample = np.random.choice(v, p=probs)
                if utility_type == 'pen_nonboolean':
                    if v_sample['gt'] and not v_sample['answer']:
                        r = penalty_fn
                    elif v_sample['gt'] and v_sample['answer']:
                        r = reward_tp
                    elif not v_sample['gt'] and v_sample['answer']:
                        r = penalty_fp
                    else:
                        r = reward_tn
                    # Doesn't make sense to sample boolean labels for
                    # a non-boolean question.
                    v_sample = None
                else:
                    p_obs = 0
                    o = v_sample['answer']
                    for a in (0, 1):
                        # Sum out variable for true answer.
                        p_obs += self.p_joint(p_r, p_slip, p_guess, prior, a, o)
                    posterior = self.p_joint(
                        p_r, p_slip, p_guess, prior, 1, o) / p_obs
                    r = reward_new_posterior(prior, posterior, utility_type,
                                             penalty_fp=penalty_fp,
                                             penalty_fn=penalty_fn,
                                             reward_tp=reward_tp,
                                             reward_tn=reward_tn)
            else:  # Expected reward.
                v_sample = None
                r = 0
                if utility_type == 'pen_nonboolean':
                    for o in (0, 1):
                        for a in (0, 1):
                            reward = reward_tp if a == o else penalty_fp
                            r += self.p_joint(p_r, p_slip, p_guess,
                                              prior, a, o) * reward
                else:
                    for o in (0, 1):
                        p_obs = 0
                        for a in (0, 1):
                            # Sum out variable for true answer.
                            p_obs += self.p_joint(p_r, p_slip,
                                                  p_guess, prior, a, o)
                        posterior = self.p_joint(
                            p_r, p_slip, p_guess, prior, 1, o) / p_obs
                        reward = reward_new_posterior(prior, posterior,
                                                      utility_type,
                                                      penalty_fp=penalty_fp,
                                                      penalty_fn=penalty_fn,
                                                      reward_tp=reward_tp,
                                                      reward_tn=reward_tn)
                        r += p_obs * reward
            metadata['rewards'].append(r)
            if v_sample is not None:
                metadata['gt'].append(v_sample['gt'])
                metadata['answer'].append(v_sample['answer'])
        return sum(metadata['rewards']), metadata

    def is_reachable(self, next_state, exp=False):
        """Return whether the state is reachable, with or without explaining."""
        if not self.is_quiz() and exp:
            raise Exception("Can't explain from non-quiz state")

        skills_learned = self.skills_learned(next_state)
        skills_lost = self.skills_lost(next_state)
        if exp and len(skills_learned) == 1:
            # Can only learn explained skill.
            return self.quiz_val == skills_learned[0]
        elif exp and len(skills_learned) == 0:
            # Cannot lose explained skill.
            return self.quiz_val not in skills_lost
        else:
            return len(skills_learned) == 0

    def __str__(self):
        if self.term:
            return 'TERM'
        s = 's'
        s += ''.join(str(int(x)) for x in self.skills)
        s += 'w{}'.format(self.worker_class)
        if self.quiz_val is not None:
            s += 'q{}'.format(self.quiz_val)
        return s

    def __eq__(self, s):
        return ((self.term and s.term) or
                (self.skills == s.skills and
                 self.quiz_val == s.quiz_val and
                 self.worker_class == s.worker_class))


def reward_new_posterior(
        prior, posterior, utility_type='pen',
        penalty_fp=-2, penalty_fn=-2, reward_tp=1, reward_tn=1):
    """Return reward of new posterior.

    Args:
        prior:          Prior probability.
        posterior:      Posterior probability.
        utility_type:   Either 'acc' (accuracy) or 'pen' (penalty).
        penalty_fp:     False positive penalty.
        penalty_fn:     False negative penalty.
        reward_tp:      True positive reward.
        reward_tn:      True negative reward.

    Returns:
        r:  Expected reward

    >>> round(reward_new_posterior(0.5, 0.7, utility_type='acc'), 6)
    0.2
    >>> round(reward_new_posterior(0.5, 0.7, utility_type='pen_diff', penalty_fp=0, penalty_fn=0), 6)
    0.2
    >>> round(reward_new_posterior(0.5, 0.7, utility_type='pen_diff', penalty_fp=-1, penalty_fn=-1), 6)
    0.4
    >>> round(reward_new_posterior(0.5, 0.7, utility_type='pen', penalty_fp=0, penalty_fn=0), 6)
    0.7
    >>> round(reward_new_posterior(0.5, 0.7, utility_type='pen', penalty_fp=-1, penalty_fn=-1), 6)
    0.4

    """
    f = lambda p: (1 - p) * reward_tn + p * penalty_fn if p <= 0.5 else \
        p * reward_tp + (1 - p) * penalty_fp
    if utility_type == 'acc':
        # Accuracy gain.
        return max(posterior, 1 - posterior) - max(prior, 1 - prior)
    elif utility_type == 'pen':
        return f(posterior)
    elif utility_type == 'pen_diff':
        return f(posterior) - f(prior)
    else:
        raise ValueError('Unexpected utility type')
