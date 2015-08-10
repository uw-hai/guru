"""Helper functions for specifying a work-learn problem.

Port from WorkLearnProblem.hpp

"""

from __future__ import division
import itertools

NINF = -999
# TODO: Change quiz_val to rule / skill.
class Action:
    def __init__(self, name, quiz_val=None):
        self.name = name
        self.quiz_val = quiz_val
    def is_quiz(self):
        return self.name == 'ask' and self.quiz_val is not None
    def uses_gold(self):
        """Return whether an action uses a gold question.
        
        Does not count explain actions as using gold, since they use the
        same gold question as the preceding ask.

        """
        return self.name == 'tell' or (self.name == 'ask' and
                                       self.quiz_val is not None)
    def __str__(self):
        s = self.name
        if self.quiz_val is not None:
            s += '-rule_{}'.format(self.quiz_val)
        return s
    def __eq__(self, a):
        return self.name == a.name and self.quiz_val == a.quiz_val

def actions_all(n_skills, tell=False, exp=False):
    """Return all actions

    Args:
        n_skills (int):   Number of skills.
        tell (bool):      Include tell actions.
        exp (bool):       Include explain action.

    """
    actions = [Action('boot'), Action('ask', None)] + \
              [Action('ask', i) for i in xrange(n_skills)]
    if exp:
        actions.append(Action('exp'))
    if tell:
        actions += [Action('tell', i) for i in xrange(n_skills)]
    return actions

# New observations ['yes', 'no'] for ask.
#observations = ['yes', 'no', 'wrong', 'right', 'term']
observations = ['wrong', 'right', 'term']

def states_all(n_skills, n_worker_classes):
    skill_values = list(itertools.product((True, False), repeat=n_skills))
    quiz_values = [None] + range(n_skills)
    worker_class_values = range(n_worker_classes)
    states_except_term = [
        State(skills=s, quiz_val=q, worker_class=w) for s, q, w in
        itertools.product(skill_values, quiz_values, worker_class_values)]
    return [State(term=True)] + states_except_term

class State:
    def __init__(self, term=None, skills=[], quiz_val=None, worker_class=None):
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
        return [i for i,(x,y) in
                enumerate(zip(self.skills, next_state.skills)) if not x and y]

    def skills_lost(self, next_state):
        return [i for i,(x,y) in
                enumerate(zip(self.skills, next_state.skills)) if x and not y]

    def has_same_skills(self, next_state):
        return self.skills == next_state.skills

    def p_has_skills(self, rule_probabilities):
        """Probability of having necessary skills"""
        if self.term:
            raise Exception('Unexpected terminal state')
        p_has_skills = 1
        for i,p in enumerate(rule_probabilities):
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
        """Probability of answering correctly"""
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

    def rewards_ask(self, p_r, p_slip, p_guess, prior, utility_type):
        # TODO: Move to separate class?
        """Expected rewards

        Args:
            utility_type: 'acc' or 'posterior'

        """
        r = 0
        for o in (0,1):
            p_obs = 0
            for a in (0, 1):
                # Sum out variable for true answer.
                p_obs += self.p_joint(p_r, p_slip, p_guess, prior, a, o)
            posterior = self.p_joint(p_r, p_slip, p_guess, prior, 1, o) / p_obs 
            # Expected reward.
            r += p_obs * reward_new_posterior(prior, posterior, utility_type)
        return r

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


def reward_new_posterior(prior, posterior, utility_type='acc'):
    """Return reward of new posterior"""
    if utility_type == 'posterior':
        if (
            (prior >= 0.5 and posterior >= 0.5) or
            (prior < 0.5 and posterior < 0.5)):
            return 0.0
        return abs(prior - posterior)
    else:
        # Chris style.
        return max(posterior, 1-posterior) - max(prior, 1-prior)

