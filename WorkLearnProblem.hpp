#include <iostream>  /* cout */
#include <limits>
#include <stdexcept>   /* invalid_argument */
#include <math.h>    /* pow */

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/RLModel.hpp>

// Most negative cost (to prevent actions from happening).
constexpr float NINF = -999;

enum {
    A_ASK = 0,
    A_EXP = 1,
    A_NOEXP = 2,
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
    };

    bool is_term() const {
        return term_;
    };

    bool has_skill(size_t skill) const {
        if (term_) {
            throw std::invalid_argument( "Terminal state has no skill" );
        }
        return bool(state_[skill]);
    };

    int quiz_val() const {
        if (term_) {
            throw std::invalid_argument( "Terminal state has no quiz val" );
        }
        return state_[n_skills_];
    };

    bool is_quiz() const {
       if (is_term()) {
           return false;
       } else {
           return quiz_val() != 0;
       }
    };

    bool is_valid_action(size_t a) const {
        bool valid_from_quiz = (a == A_EXP || a == A_NOEXP);
        if (is_term()) {
            return false;
        } else if (is_quiz()) {
            return valid_from_quiz;
        } else {
            // Mutually exclusive.
            return !valid_from_quiz;
        }
    };

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
    };

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

/* Helper functions */
// Reward of new posterior.
double rewardNewPosterior(const double prior, const double posterior) {
    if ((prior >= 0.5 && posterior >= 0.5) || (prior < 0.5 && posterior < 0.5)) {
        return 0.0;
    }
    return std::abs(prior - posterior);
}

double pHasSkills(const WorkerState& st, const std::vector<double>& p_rule) {
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
    return p_has_skills; 
}

// Probability of answering correctly.
double pRight(const WorkerState& st, const std::vector<double>& p_rule, const double p_slip, const double p_guess) {
    double p_has_skills = pHasSkills(st, p_rule);
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

double rewardsAsk(const WorkerState& st, const std::vector<double>& p_r, const double p_slip, const double p_guess, const double p_1) {
    double r = 0;
    for ( int obsVal = 0; obsVal < 2; ++obsVal ) {
        double pObs = pJoint(st, p_r, p_slip, p_guess, p_1, 0, obsVal) +
                      pJoint(st, p_r, p_slip, p_guess, p_1, 1, obsVal);

        double posterior = pJoint(st, p_r, p_slip, p_guess, p_1, 1, obsVal) / pObs;

        r += pObs * rewardNewPosterior(p_1, posterior);
    }
    return r;
} 


std::tuple<AIToolbox::POMDP::Model<AIToolbox::MDP::Model>,
           AIToolbox::POMDP::RLModel<AIToolbox::MDP::Model> >
           makeWorkLearnProblem(const double cost, const double cost_exp, const double cost_living, const double p_learn, const double p_leave, const double p_slip, const double p_guess, const std::vector<double> p_r, const double p_1, const size_t n_skills, const size_t S, AIToolbox::POMDP::Experience* const exp) {

    size_t A = n_skills + N_RES_A;

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::Table3D transitions(boost::extents[S][A][S]);
    AIToolbox::Table3D rewards(boost::extents[S][A][S]);
    AIToolbox::Table3D observations(boost::extents[S][A][O]);

    boost::multi_array<bool, 3> transitions_clamped(boost::extents[S][A][S]);
    boost::multi_array<int, 2> transitions_shared_group(boost::extents[S][A]);
    boost::multi_array<int, 3> transitions_shared_order(boost::extents[S][A][S]);
    boost::multi_array<bool, 3> observations_clamped(boost::extents[S][A][O]);
    boost::multi_array<int, 2> observations_shared_group(boost::extents[S][A]);
    boost::multi_array<int, 3> observations_shared_order(boost::extents[S][A][O]);
    std::fill(transitions_clamped.origin(), transitions_clamped.origin() + transitions_clamped.num_elements(), true);
    std::fill(transitions_shared_group.origin(), transitions_shared_group.origin() + transitions_shared_group.num_elements(), -1);
    std::fill(transitions_shared_order.origin(), transitions_shared_order.origin() + transitions_shared_order.num_elements(), -1);
    std::fill(observations_clamped.origin(), observations_clamped.origin() + observations_clamped.num_elements(), true);
    std::fill(observations_shared_group.origin(), observations_shared_group.origin() + observations_shared_group.num_elements(), -1);
    std::fill(observations_shared_order.origin(), observations_shared_order.origin() + observations_shared_order.num_elements(), -1);

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
            for ( size_t a = 0; a < A; ++a )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    transitions_clamped[s][a][s1] = true;
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
                    transitions_clamped[s][action_index(a)][s1] = false;
                    transitions_shared_group[s][action_index(a)] = 0;
                    transitions_shared_order[s][action_index(a)][s1] = 0;
                    transitions[s][action_index(a)][s] = 0.0;
                }
                transitions[s][A_ASK][s1] = p_leave;
                transitions_clamped[s][A_ASK][s1] = false;
                transitions_shared_group[s][A_ASK] = 0;
                transitions_shared_order[s][A_ASK][s1] = 0;
                transitions[s][A_ASK][s] = 1.0 - p_leave;
                transitions_clamped[s][A_ASK][s] = false;
                transitions_shared_group[s][A_ASK] = 0;
                transitions_shared_order[s][A_ASK][s] = 1;
                // Reward for asking a question.
                rewards[s][A_ASK][s] += cost;
                rewards[s][A_ASK][s] += rewardsAsk(st, p_r, p_slip, p_guess, p_1);
            } else if (st1.is_term()) {
                // Done with terminal state.
                // IMPORTANT: We don't allow booting from quiz states.
                continue;
            } else if (stateCompare(st, st1) == -1 /* same skills */ &&
                       st.quiz_val() == 0 && st1.quiz_val() != 0) {
                // Quizzing takes to quiz state with same latent skills.
                size_t sk = st1.quiz_val() - 1;
                transitions[s][action_index(sk)][s1] = 1.0 - p_leave;
                transitions_clamped[s][action_index(sk)][s1] = false;
                transitions_shared_group[s][action_index(sk)] = 0;
                transitions_shared_order[s][action_index(sk)][s1] = 1;
                rewards[s][action_index(sk)][s1] += cost;
            } else if (st.quiz_val() != 0 && st1.quiz_val() == 0 ) {
                // Explaining happens from quiz state to non-quiz state.
                size_t sk = st.quiz_val() - 1;
                int compV = stateCompare(st, st1);
                if (compV == -1 /* same skills */ && !st.has_skill(sk)) {
                    transitions[s][A_EXP][s1] = 1.0 - p_learn;
                    transitions_clamped[s][A_EXP][s1] = false;
                    transitions_shared_group[s][A_EXP] = 1;
                    transitions_shared_order[s][A_EXP][s1] = 1;
                    transitions[s][A_EXP][s] = 0.0;
                } else if (compV == static_cast<int>(sk)) {
                    transitions[s][A_EXP][s1] = p_learn;
                    transitions_clamped[s][A_EXP][s1] = false;
                    transitions_shared_group[s][A_EXP] = 1;
                    transitions_shared_order[s][A_EXP][s1] = 0;
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

            int has_skills = pHasSkills(st, p_rule_gold) == 1.0;
            double pr = pRight(st, p_rule_gold, p_slip, p_guess);
            observations[s][action_index(sk)][O_RIGHT] = pr;
            observations_shared_group[s][action_index(sk)] = has_skills;
            observations_shared_order[s][action_index(sk)][O_RIGHT] = 0;
            observations_clamped[s][action_index(sk)][O_RIGHT] = false;
            observations[s][action_index(sk)][O_WRONG] = 1.0 - pr;
            observations_clamped[s][action_index(sk)][O_WRONG] = false;
            observations_shared_group[s][action_index(sk)] = has_skills;
            observations_shared_order[s][action_index(sk)][O_WRONG] = 1;
            observations[s][action_index(sk)][O_TERM] = 0.0;
        }
    }

    // Initial beliefs.
    std::vector<double> initial_belief;
    std::vector<double> initial_belief_clamped;
    for ( size_t s = 0; s < S; ++s ) {
        WorkerState st(s, n_skills);
        if (st.is_term()) {
            initial_belief_clamped.push_back(1);
            initial_belief.push_back(0.0);
        } else if (st.quiz_val() != 0) {
            initial_belief_clamped.push_back(1);
            initial_belief.push_back(0.0);
        } else {
            initial_belief_clamped.push_back(0);
            initial_belief.push_back(1.0 / std::pow(2, n_skills));
        }
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    AIToolbox::POMDP::RLModel<AIToolbox::MDP::Model> RLmodel(exp, model);
    RLmodel.setInitialBeliefFunction(initial_belief);
    RLmodel.setInitialBeliefClampFunction(initial_belief_clamped);
    RLmodel.setTransitionClampFunction(transitions_clamped);
    RLmodel.setTransitionSharedGroupFunction(transitions_shared_group);
    RLmodel.setTransitionSharedOrderFunction(transitions_shared_order);
    RLmodel.setObservationClampFunction(observations_clamped);
    RLmodel.setObservationSharedGroupFunction(observations_shared_group);
    RLmodel.setObservationSharedOrderFunction(observations_shared_order);

    RLmodel.setInitialBeliefPriorFunction(0.1);
    RLmodel.setTransitionPriorFunction(0.1);
    RLmodel.setObservationPriorFunction(0.1);

    // BUG: Move to test.
    // Print transitions & rewards.
    for ( size_t s = 0; s < S; ++s )
        for ( size_t a = 0; a < A; ++a )
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                auto ws = WorkerState(s, n_skills);
                auto ws1 = WorkerState(s1, n_skills);
                std::cout << "|" << ws << "|" << " . " << a << " . " << "|" << ws1 << "|" << " -> " << model.getTransitionProbability(s, a, s1) << " (" << model.getExpectedReward(s, a, s1) << ")";
                if (RLmodel.isTransitionClamped(s, a, s1)) {
                    std::cout << " C";
                }
                std::cout << std::endl;
            }

    // Print observations.
    for ( size_t s1 = 0; s1 < S; ++s1 )
        for ( size_t a = 0; a < A; ++a )
            for ( size_t o = 0; o < O; ++o ) {
                auto ws1 = WorkerState(s1, n_skills);
                std::cout << "|" << ws1 << "|" << " . " << a << " . " << "|" << o << "|" << " -> " << model.getObservationProbability(s1, a, o);
                if (RLmodel.isObservationClamped(s1, a, o)) {
                    std::cout << " C";
                }
                std::cout << std::endl;
            }

    return std::make_tuple(model, RLmodel);
}
