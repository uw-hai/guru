#include <iostream>  /* cout */
#include <limits>
#include <stdexcept>   /* invalid_argument */
#include <math.h>    /* pow */

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

// Most negative cost (to prevent actions from happening).
constexpr float NINF = -999;

class WorkerState {
  public:
    enum {
        TERM = 0,
        N_RES_S = 1   /* Number of reserved states */
    };

    // Initializes state. The number of possible states S_ is equal to the
    // - terminal state, plus
    // - quiz state for each skill (or no skill), plus
    // - latent bit for each skill.
    WorkerState(const size_t encoded_state, const size_t n_skills) : n_skills_(n_skills), S_(1 + (n_skills + 1) * std::pow(2, n_skills)), term_(encoded_state == TERM) {
        if (!term_) {
            decodeState(encoded_state - N_RES_S);
        }
    }

    bool is_term() const {
        return term_;
    }

    bool has_skill(size_t skill) const {
        if (term_) {
            throw std::invalid_argument( "Terminal state has no skill" );
        }
        return bool(state_[skill]);
    }

    int quiz_val() const {
        if (term_) {
            throw std::invalid_argument( "Terminal state has no quiz val" );
        }
        return state_[n_skills_];
    }

    const size_t n_skills_;
  private:
    // Decodes a state s into an array of type
    // [S_0 (2), S_1 (2), ..., S_N (2), QUIZ (N_SKILLS + 1)]
    void decodeState(size_t s) {
        size_t length = n_skills_ + 1;
        size_t divisor;
        for ( size_t i = 0; i < length; ++i ) {
            if (i == 0) {
                divisor = n_skills_ + 1;
            } else {
                divisor = 2;
            }
            state_.push_back(s % divisor);
            s /= divisor;
        }
        // Put array in standard written form (most significant bit at left).
        std::reverse(state_.begin(), state_.end());
    }

    const size_t S_;
    const bool term_;
    std::vector<int> state_;
};

std::ostream &operator<<(std::ostream &os, const WorkerState &st) { 
    if (st.is_term()) {
        return os << "TERM";
    }
    std::stringstream out;
    for ( size_t i = 0; i < st.n_skills_; ++i ) {
        os << st.has_skill(i) << " ";
    }
    os << "q" << st.quiz_val();
    return os;
}

enum {
    A_EXP = 0,
    A_NOEXP = 1,
    A_ASK = 2,
    A_BOOT = 3,
    N_RES_A = 4
    /* Quiz actions: 4 to N_SKILLS + 3 */
};
// Returns the correct action index.
size_t action_index(const size_t a) {
    return N_RES_A + a;
}

size_t O = 3;
enum {
    O_WRONG = 0,
    O_RIGHT= 1,
    O_TERM = 2
};

/* Helper functions */
// Reward of new posterior.
double rewardNewPosterior(const double prior, const double posterior) {
    if ((prior >= 0.5 && posterior >= 0.5) || (prior < 0.5 && posterior < 0.5)) {
        return 0.0;
    }
    return std::abs(prior - posterior);
}

// Probability of answering correctly.
double pRight(const WorkerState& st, const std::vector<double>& p_rule, const double p_slip, const double p_guess) {
    size_t n_skills = p_rule.size();
    if (st.is_term()) {
        throw std::invalid_argument( "Received terminal state" );
    }
    double p_has_skills = 1.0;
    for ( size_t i = 0; i < n_skills; ++i ) {
        if (!st.has_skill(i)) {
            p_has_skills *= 1.0 - p_rule[i];
        }
    }

    return p_has_skills * (1.0 - p_slip) + (1 - p_has_skills) * p_guess;
}

// Joint probability.
double pJoint(const WorkerState& st, const std::vector<double>& p_rule, const double p_slip, const double p_guess, const double priorProb, const int priorVal, const int obsVal) {
    double p1, p2;
    if (priorVal == 0) {
        p1 = 1.0 - priorProb;
    } else {
        p1 = priorProb;
    }

    double pr = pRight(st, p_rule, p_slip, p_guess);
    if (obsVal == priorVal) {
        p2 = pr;
    } else {
        p2 = (1.0 - pr);
    }

    return p1 * p2;
}

// Calculate relationship between latent skills for states.
// Return value:
//      0 to N_SKILLS - 1:  Skill bit that has been flipped positive
//      -1:                 States are equal
//      -2:                 States are unequal by more than one bit flipped positive
int stateCompare(const WorkerState& st, const WorkerState& st1) {
    if (st.is_term() || st1.is_term()) {
        throw std::invalid_argument( "Unexpected terminal state" );
    } else if (st.n_skills_ != st1.n_skills_) {
        throw std::invalid_argument( "Unequal number of skills" );
    }
    int skill_flipped = -1;
    for ( size_t i = 0; i < st.n_skills_; ++i ) {
        if (!st.has_skill(i) && st1.has_skill(i) && skill_flipped < 0) {
            skill_flipped = i;
        } else if (st.has_skill(i) != st1.has_skill(i)) {
            return -2;
        } 
    }
    return skill_flipped;    
}

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeWorkLearnProblem(const double cost, const double cost_exp, const double cost_living, const double p_learn, const double p_leave, const double p_slip, const double p_guess, const std::vector<double> p_r, const double p_1, const size_t n_skills, const size_t S) {

    size_t A = n_skills + N_RES_A;

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::Table3D transitions(boost::extents[S][A][S]);
    AIToolbox::Table3D rewards(boost::extents[S][A][S]);
    AIToolbox::Table3D observations(boost::extents[S][A][O]);

    // Initialize tables.
    // Assumes all tables initialized with 0s by default.
    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            transitions[s][a][s] = 1.0;
    // All observations equally likely, except for terminal state observation.
    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            for ( size_t o = 0; o < O-1; ++o )
                observations[s][a][o] = 1.0 / (O-1);
    /* Penalty for staying alive (transitions from non-terminal states).
     * Note:    Not necessary now that we effectively ban some actions
     *          depending on whether or not the state is a quiz state.
     */
    for ( size_t s = 0; s < S; ++s ) {
        auto st = WorkerState(s, n_skills);
        if (!st.is_term()) {
            for ( size_t a = 0; a < A; ++a )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    rewards[s][a][s1] = cost_living;
        }
    }

    // Transitions
    for ( size_t s = 0; s < S; ++s ) {
        auto st = WorkerState(s, n_skills);
        if (st.is_term()) {
            continue;
        } else if (st.quiz_val() == 0) {
            // Not allowed to explain from non-quiz state.
            rewards[s][A_EXP][s] = NINF;
            rewards[s][A_NOEXP][s] = NINF;
        } else {
            // Must be a quiz state.
            // Not allowed to quiz, boot, or ask from a quiz state.
            for ( size_t a = 0; a < n_skills; ++a ) {
                rewards[s][action_index(a)][s] = NINF;
            }
            rewards[s][A_BOOT][s] = NINF;
            rewards[s][A_ASK][s] = NINF;
        }
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            auto st1 = WorkerState(s1, n_skills);
            if (st.quiz_val() == 0 && st1.is_term()) {
                // Executed once for each non-quiz starting state.

                // Booting takes to terminal state.
                transitions[s][A_BOOT][s1] = 1.0;
                transitions[s][A_BOOT][s] = 0.0;
                // Worker might leave when we quiz or ask.
                for ( size_t a = 0; a < n_skills; ++a ) {
                    transitions[s][action_index(a)][s1] = p_leave;
                    transitions[s][action_index(a)][s] = 0.0;
                }
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
            } else if (st1.is_term()) {
                // Done with terminal state.
                // IMPORTANT: We don't allow booting from quiz states.
                continue;
            } else if (stateCompare(st, st1) == -1 /* same skills */ &&
                       st.quiz_val() == 0 && st1.quiz_val() != 0) {
                // Quizzing takes to quiz state with same latent skills.
                size_t sk = st1.quiz_val() - 1;
                transitions[s][action_index(sk)][s1] = 1.0 - p_leave;
                rewards[s][action_index(sk)][s1] += cost;
            } else if (st.quiz_val() != 0 && st1.quiz_val() == 0 ) {
                // Explaining happens from quiz state to non-quiz state.
                size_t sk = st.quiz_val() - 1;
                int compV = stateCompare(st, st1);
                if (compV == -1 /* same skills */ && !st.has_skill(sk)) {
                    transitions[s][A_EXP][s1] = 1.0 - p_learn;
                    transitions[s][A_EXP][s] = 0.0;
                } else if (compV == static_cast<int>(sk)) {
                    transitions[s][A_EXP][s1] = p_learn;
                } else if (compV == -1 /* same_skills */ && st.has_skill(sk)) {
                    transitions[s][A_EXP][s1] = 1.0;
                    transitions[s][A_EXP][s] = 0.0;
                }
                rewards[s][A_EXP][s1] += cost_exp;

                if (compV == -1 /* same skills */) {
                    transitions[s][A_NOEXP][s1] = 1.0;
                    transitions[s][A_NOEXP][s] = 0.0;
                }
            }
        }
    }

    // Observations.
    for ( size_t s = 0; s < S; ++s ) {
        auto st = WorkerState(s, n_skills);
        if (st.is_term()) {
            // Always know when we enter terminal state.
            for ( size_t a = 0; a < A; ++a ) {
                observations[s][a][O_RIGHT] = 0.0;
                observations[s][a][O_WRONG] = 0.0;
                observations[s][a][O_TERM] = 1.0;
            }
        } else if (st.quiz_val() != 0) {
            // Assume that teaching actions ask questions that require only
            // the skill being taught.
            size_t sk = st.quiz_val() - 1;
            std::vector<double> p_rule_gold;
            for ( size_t i = 0; i < n_skills; ++i ) {
                if (i == sk) {
                    p_rule_gold.push_back(1.0); 
                } else {
                    p_rule_gold.push_back(0.0);
                }
            }

            double pr = pRight(st, p_rule_gold, p_slip, p_guess);
            observations[s][action_index(sk)][O_RIGHT] = pr;
            observations[s][action_index(sk)][O_WRONG] = 1.0 - pr;
            observations[s][action_index(sk)][O_TERM] = 0.0;
        }
    }

    // Print transitions & rewards.
    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            for ( size_t s1 = 0; s1 < S; ++s1 )
                std::cout << "|" << WorkerState(s, n_skills) << "|" << " . " << a << " . " << "|" << WorkerState(s1, n_skills) << "|" << " -> " << transitions[s][a][s1] << " (" << rewards[s][a][s1] << ")" << std::endl;

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    return model;
}
