#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <limits>

/* Problem encoding */
constexpr int MAX_DIM = 2;

using StateType = std::array<int, 3>;
size_t S = 5;
enum {
    SKILL_0 = 0,
    IS_QUIZ = 1,
    IS_TERM = 2
};

size_t encodeState(const StateType& state) {
    size_t n = 0; unsigned multiplier = 1;
    for ( auto s : state ) {
        n += multiplier * s;
        multiplier *= MAX_DIM;
    }
    // All states with TERM = 1 are equivalent.
    return std::min(n, S-1);
}

StateType decodeState(size_t n) {
    StateType state;
    for ( auto &s : state ) {
        s = n % MAX_DIM;
        n /= MAX_DIM;
    }
    return state;
}

size_t A = 4;
enum {
    A_QUIZ_0 = 0,
    A_EXP_0 = 1,
    A_ASK = 2,
    A_BOOT = 3
};
std::string action_name(size_t a) {
    if (a == A_QUIZ_0) {
        return "A_QUIZ";
    } else if (a == A_EXP_0) {
        return "A_EXP";
    } else if (a == A_ASK) {
        return "A_ASK";
    } else {
        return "A_BOOT";
    }
}

size_t O = 3;
enum {
    O_WRONG = 0,
    O_RIGHT= 1,
    O_TERM = 2
};
std::string observation_name(size_t o) {
    if (o == O_WRONG) {
        return "O_WRONG";
    } else if (o == O_RIGHT) {
        return "O_RIGHT";
    } else {
        return "O_TERM";
    }
}


/* Helper functions */
// Reward of new posterior.
double rewardNewPosterior(const double prior, const double posterior) {
    if ((prior >= 0.5 && posterior >= 0.5) || (prior < 0.5 && posterior < 0.5)) {
        return 0.0;
    }
    return std::abs(prior - posterior);
}

// Probability of answering correctly
double pRight(const StateType& s, const double ruleProb, const double p_slip, const double p_guess) {
    if (!s[SKILL_0]) {
        return (1.0 - ruleProb) * (1.0 - p_slip) + ruleProb * p_guess;
    } else {
        return 1.0 - p_slip;
    }
}

// Joint probability.
double pJoint(const StateType& s, const double ruleProb, const double p_slip, const double p_guess, const double priorProb, const int priorVal, const int obsVal) {
    double p1, p2;
    if (priorVal == 0) {
        p1 = 1.0 - priorProb;
    } else {
        p1 = priorProb;
    }

    double pr = pRight(s, ruleProb, p_slip, p_guess);
    if (obsVal == priorVal) {
        p2 = pr;
    } else {
        p2 = (1.0 - pr);
    }

    return p1 * p2;
}

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeWorkLearnProblem(const double cost, const double cost_living, const double p_learn, const double p_leave, const double p_slip, const double p_guess, const double p_r, const double p_1) {
    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::Table3D transitions(boost::extents[S][A][S]);
    AIToolbox::Table3D rewards(boost::extents[S][A][S]);
    AIToolbox::Table3D observations(boost::extents[S][A][O]);

    // Initialize tables.
    // Assume all tables initialized with 0s by default.
    // BUG: Check this assumption.
    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            transitions[s][a][s] = 1.0;
    // All observations equally likely, except for terminal state observation.
    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            for ( size_t o = 0; o < O-1; ++o )
                observations[s][a][o] = 1.0 / (O-1);
    // Small penalty for staying alive (transitions from non-terminal states).
    for ( size_t s = 0; s < S; ++s ) {
        StateType st = decodeState(s);
        if (!st[IS_TERM]) {
            for ( size_t a = 0; a < A; ++a )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    rewards[s][a][s1] = cost_living;
        }
    }


    // Transitions
    for ( size_t s = 0; s < S; ++s ) {
        StateType st = decodeState(s);
        if (st[IS_TERM]) {
            continue;
        } else if (!st[IS_QUIZ]) {
            // Not allowed to explain from non-quiz state.
            // rewards[s][A_EXP_0][s] = std::numeric_limits<double>::lowest();
            rewards[s][A_EXP_0][s] = -999;
        } else {
            // Must be a quiz state.
            // Not allowed to quiz or boot from a quiz state.
            // rewards[s][A_QUIZ_0][s] = std::numeric_limits<double>::lowest();
            // rewards[s][A_BOOT][s] = std::numeric_limits<double>::lowest();
            rewards[s][A_QUIZ_0][s] = -999;
            rewards[s][A_BOOT][s] = -999;
        }
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            StateType st1 = decodeState(s1);

            if (!st[IS_QUIZ] && st1[IS_TERM]) {
                // Executed once for each non-quiz starting state.

                // Booting takes to terminal state.
                transitions[s][A_BOOT][s1] = 1.0;
                transitions[s][A_BOOT][s] = 0.0;
                // Worker might leave when we quiz or ask.
                transitions[s][A_QUIZ_0][s1] = p_leave;
                transitions[s][A_QUIZ_0][s] = 0.0;
                transitions[s][A_ASK][s1] = p_leave;
                transitions[s][A_ASK][s] = 1.0 - p_leave;
                // Reward for asking a question.
                rewards[s][A_ASK][s] += cost;
                for ( int obsVal = 0; obsVal < 2; ++obsVal ) {
                    double pObs = pJoint(st, p_r, p_slip, p_guess, p_1, 0, obsVal) +
                                  pJoint(st, p_r, p_slip, p_guess, p_1, 1, obsVal);

                    double posterior = pJoint(st, p_r, p_slip, p_guess, p_1, 1, obsVal) / pObs;

                    rewards[s][A_ASK][s] += pObs * rewardNewPosterior(p_1, posterior);
                }
            } else if (st1[IS_TERM]) {
                // Done with terminal state.
                // IMPORTANT: We don't allow booting from the quiz state.
                continue;
            } else if (st[SKILL_0] == st1[SKILL_0] &&
                       !st[IS_QUIZ] && st1[IS_QUIZ]) {
                // Quizzing takes to special quiz state.
                transitions[s][A_QUIZ_0][s1] = 1.0 - p_leave;
                rewards[s][A_QUIZ_0][s1] += cost;
            } else if (st[IS_QUIZ] && !st1[IS_QUIZ] ) {
                // Explaining happens from quiz state.
                if (!st[SKILL_0] && !st1[SKILL_0]) {
                    transitions[s][A_EXP_0][s1] = 1.0 - p_learn;
                    transitions[s][A_EXP_0][s] = 0.0;
                } else if (!st[SKILL_0] && st1[SKILL_0]) {
                    transitions[s][A_EXP_0][s1] = p_learn;
                } else if (st[SKILL_0] && st1[SKILL_0]) {
                    transitions[s][A_EXP_0][s1] = 1.0;
                    transitions[s][A_EXP_0][s] = 0.0;
                }
            }
        }
    }

    // Observations.
    for ( size_t s = 0; s < S; ++s ) {
        StateType st = decodeState(s);
        if (st[IS_TERM]) {
            observations[s][A_QUIZ_0][O_RIGHT] = 0.0;
            observations[s][A_QUIZ_0][O_WRONG] = 0.0;
            observations[s][A_QUIZ_0][O_TERM] = 1.0;
            observations[s][A_ASK][O_RIGHT] = 0.0;
            observations[s][A_ASK][O_WRONG] = 0.0;
            observations[s][A_ASK][O_TERM] = 1.0;
            observations[s][A_BOOT][O_RIGHT] = 0.0;
            observations[s][A_BOOT][O_WRONG] = 0.0;
            observations[s][A_BOOT][O_TERM] = 1.0;
        } else if (st[IS_QUIZ]) {
            // Assume that teaching actions always use the rule.
            double pr = pRight(st, 1.0, p_slip, p_guess);
            observations[s][A_QUIZ_0][O_RIGHT] = pr;
            observations[s][A_QUIZ_0][O_WRONG] = 1.0 - pr;
            observations[s][A_QUIZ_0][O_TERM] = 0.0;
        }
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    /*
    // Check isTerminal correctness.
    for ( size_t s = 0; s < S; ++s) {
        std::cout << s << ": " << model.isTerminal(s) << std::endl;
    }
    */

    return model;
}
