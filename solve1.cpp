/* BUG: Description */
#include <iostream>
#include <iomanip>  /* string padding */
#include <fstream>
#include <random>
#include <map>
#include <time.h>  /* time_t, time, ctime */

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/RLModel.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include "WorkLearnProblem.hpp"

const size_t PBVI_NBELIEFS = 1000;
const double PBVI_EPSILON = 0.01;

// Make experiments deterministic.
const int SEED = 0;

namespace po = boost::program_options;
using namespace AIToolbox;
using boost::property_tree::ptree;

// Useful for parsing ptree arrays.
template <typename T>
std::vector<T> as_vector(ptree const& pt, ptree::key_type const& key)
{
    std::vector<T> r;
    for (auto& item : pt.get_child(key))
        r.push_back(item.second.get_value<T>());
    return r;
}

// Checks if belief state is quiz or non-quiz.
bool is_quiz(const POMDP::Belief& b, const size_t n_skills, const size_t S) {
    for ( size_t s = 0 ; s < S; ++s ) {
        auto st = WorkerState(s, n_skills);
        if (st.is_term()) {
            continue;
        } else if (st.quiz_val() == 0) {
            return b[s] == 0.0;
        } else {
            return b[s] > 0.0;
        }
    }
    return false;  // should never get here
}

int main(int argc, char **argv) {
    // Parse input.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("input,i", po::value<std::string>()->required(), "Config file")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 
    std::string config = vm["input"].as<std::string>();
    boost::filesystem::path path(config);
    const std::string exp_name = path.filename().replace_extension("").string();

    ptree pt;
    boost::property_tree::json_parser::read_json(config, pt);
    const double cost = pt.get<double>("params.cost");
    const double cost_exp = pt.get<double>("params.cost_exp");
    const double cost_living = pt.get<double>("params.cost_living");
    const double p_learn = pt.get<double>("params.p_learn");
    const double p_leave = pt.get<double>("params.p_leave");
    const double p_slip = pt.get<double>("params.p_slip");
    const double p_guess = pt.get<double>("params.p_guess");
    const std::vector<double> p_r = as_vector<double>(pt, "params.p_r");
    const std::vector<double> p_s = as_vector<double>(pt, "params.p_s");
    const double p_1 = pt.get<double>("params.p_1");
    const size_t iterations = pt.get<size_t>("params.iterations", 1);
    const size_t episodes = pt.get<size_t>("params.episodes");
    const double discount = pt.get<double>("params.discount");
    ptree &policies = pt.get_child("policies");

    // Find maximum horizons.
    /*
    size_t max_horizon_pbvi = 0;
    size_t max_horizon_ip = 0;
    for (auto& pdef : policies) {
        std::string ptype = pdef.second.get<std::string>("type");
        if (ptype == "ip") {
            max_horizon_ip = std::max(max_horizon_ip, pdef.second.get<size_t>("horizon"));
        } else if (ptype == "pbvi") {
            max_horizon_pbvi = std::max(max_horizon_pbvi, pdef.second.get<size_t>("horizon"));
        }
    }
    */

    // Process input.
    std::cout << "Loading...\n";
    size_t n_skills = p_r.size();
    std::cout << "Got size...\n";
    size_t S = (n_skills + 1) * std::pow(2, n_skills) + 1;
    std::cout << "Set S...\n";
    POMDP::Experience exp;
    std::cout << "Made exp...\n";
    auto res = makeWorkLearnProblem(cost, cost_exp, cost_living, p_learn, p_leave, p_slip, p_guess, p_r, p_1, n_skills, S, &exp);
    std::cout << "Made models...\n";
    auto model_true = std::get<0>(res);
    auto model = std::get<1>(res);
    model.setDiscount(discount);

    // Define start and terminal belief states.
    POMDP::Belief b_start;
    POMDP::Belief b_term;
    for ( size_t s = 0; s < S; ++s ) {
        auto st = WorkerState(s, n_skills);
        if (st.is_term()) {
            b_term.push_back(1.0);
        } else {
            b_term.push_back(0.0);
        }

        if (st.is_term() || st.is_quiz()) {
            b_start.push_back(0.0);
        } else {
            double pr = 1.0;
            for ( size_t i = 0; i < n_skills; ++i ) {
                if (st.has_skill(i)) {
                    pr *= p_s[i];
                } else {
                    pr *= 1 - p_s[i];
                }
            }
            b_start.push_back(pr);
        }
    }

    // Save state names.
    // TODO: Save action + transition names.
    {
        std::ofstream output("res/" + exp_name + "_names.csv");
        output << "i,type,s\n";
        for ( size_t s = 0; s < S; ++s ) {
            auto st = WorkerState(s, n_skills);
            output << s << ",";
            output << "state,";
            output << st << "\n";
        }
    }
    // Save parameters of true model.
    {
        std::ofstream output("res/" + exp_name + "_model_true.csv");
        output << "param_type,shared_g,shared_o,clamped,a1,a2,a3,v\n";
        for ( size_t s = 0; s < model.getS(); ++s )
            for ( size_t a = 0; a < model.getA(); ++a )
                for ( size_t s1 = 0; s1 < model.getS(); ++s1 ) {
                    output << "reward," << "," << "," << ","
                           << s << ","
                           << a << ","
                           << s1 << ","
                           << model_true.getExpectedReward(s, a, s1) << "\n";
                    output << "transition,"
                           << model.getTransitionSharedGroup(s, a) << ","
                           << model.getTransitionSharedOrder(s, a, s1) << ","
                           << model.isTransitionClamped(s, a, s1) << ","
                           << s << ","
                           << a << ","
                           << s1 << ","
                           << model_true.getTransitionProbability(s, a, s1) << "\n";
                }
        for ( size_t s1 = 0; s1 < model.getS(); ++s1 )
            for ( size_t a = 0; a < model.getA(); ++a )
                for ( size_t o = 0; o < model.getO(); ++o )
                    output << "observation,"
                           << model.getObservationSharedGroup(s1, a) << ","
                           << model.getObservationSharedOrder(s1, a, o) << ","
                           << model.isObservationClamped(s1, a, o) << ","
                           << s1 << ","
                           << a << ","
                           << o << ","
                           << model_true.getObservationProbability(s1, a, o) << "\n";
        for ( size_t s = 0; s < model.getS(); ++s )
            output << "initial," << "," << ","
                   << model.isInitialBeliefClamped(s) << ","
                   << s << "," << "," << ","
                   << b_start[s] << "\n";
    }

    // Make experiments reproducible. For true randomness, use:
    //      std::default_random_engine rand(Impl::Seeder::getSeed());
    std::default_random_engine rand(SEED);
    static std::uniform_real_distribution<double> uniformDistribution(0.0, 1.0);

    // Prepare to store action-level info and model estimates.
    std::ofstream output("res/" + exp_name + ".txt");
    output << "iteration,episode,t,policy,sys_t,a,s,o,r,b\n";
    std::ofstream output_model("res/" + exp_name + "_model_est.csv");
    output_model << "iteration,episode,t,policy,param_type,shared_g,shared_o,a1,a2,a3,v\n";

    // Perform experiments
    for ( size_t it = 0; it < iterations; ++it ) {
        for (auto& pdef : policies) {
            exp.reset();
            std::string pname = pdef.second.get<std::string>("name");
            std::cout << "Policy " << pname << std::endl;
            std::string ptype = pdef.second.get<std::string>("type");
            size_t h;
            if (ptype == "ip" || ptype == "pbvi") {
                h = pdef.second.get<size_t>("horizon");
            }
            boost::optional<double> epsilon = pdef.second.get_optional<double>("learn.epsilon");
            // Assume all RL policies must specify a value of epsilon.
            // BUG: RL models and non-RL models can't co-exist in a config file.
            if (epsilon) {
                model.setRandParams();
            }

            for ( size_t ep = 0; ep < episodes; ++ep ) {
                std::cout << "Episode " << ep << std::endl;
                exp.newEpisode();
                size_t s_start = sampleProbability(S, b_start, rand);
                auto st = WorkerState(s_start, n_skills);
                //std::cout << "Start state: " << st << std::endl;
                POMDP::Belief b;
                size_t s = s_start;
                size_t t = 0;

                // Estimate and solve.
                if (epsilon) {
                    model.runEM();
                    const auto ib = model.getInitialBeliefFunction();
                    b.assign(ib.origin(), ib.origin() + ib.num_elements());

                    // Store estimated model.
                    for ( size_t s = 0; s < model.getS(); ++s )
                        for ( size_t a = 0; a < model.getA(); ++a )
                            for ( size_t s1 = 0; s1 < model.getS(); ++s1 )
                                if (!model.isTransitionClamped(s, a, s1))
                                    output_model << it << ","
                                                 << ep << ","
                                                 << t << ","
                                                 << pname << ","
                                                 << "transition,"
                                                 << model.getTransitionSharedGroup(s, a) << ","
                                                 << model.getTransitionSharedOrder(s, a, s1) << ","
                                                 << s << ","
                                                 << a << ","
                                                 << s1 << ","
                                                 << model.getTransitionProbability(s, a, s1) << "\n";
                    for ( size_t s1 = 0; s1 < model.getS(); ++s1 )
                        for ( size_t a = 0; a < model.getA(); ++a )
                            for ( size_t o = 0; o < model.getO(); ++o )
                                if (!model.isObservationClamped(s1, a, o))
                                    output_model << it << ","
                                                 << ep << ","
                                                 << t << ","
                                                 << pname << ","
                                                 << "observation,"
                                                 << model.getObservationSharedGroup(s1, a) << ","
                                                 << model.getObservationSharedOrder(s1, a, o) << ","
                                                 << s1 << ","
                                                 << a << ","
                                                 << o << ","
                                                 << model.getObservationProbability(s1, a, o) << "\n";
                    for ( size_t s = 0; s < model.getS(); ++s )
                        if (!model.isInitialBeliefClamped(s))
                            output_model << it << ","
                                         << ep << ","
                                         << t << ","
                                         << pname << ","
                                         << "initial," << "," << ","
                                         << s << "," << "," << ","
                                         << model.getInitialBeliefProbability(s) << "\n";

                    // Print transitions & rewards.
                    /*
                    std::cout << "---T---" << std::endl;
                    for ( size_t s = 0; s < model.getS(); ++s )
                        for ( size_t a = 0; a < model.getA(); ++a )
                            for ( size_t s1 = 0; s1 < model.getS(); ++s1 ) {
                                auto ws = WorkerState(s, n_skills);
                                auto ws1 = WorkerState(s1, n_skills);
                                if (!model.isTransitionClamped(s, a, s1)) {
                                    std::cout << "|" << ws << "|" << " . " << a << " . " << "|" << ws1 << "|" << " -> " << model.getTransitionProbability(s, a, s1) << " (" << model.getExpectedReward(s, a, s1) << ")";
                                    std::cout << std::endl;
                                }
                            }

                    // Print observations.
                    std::cout << "---O---" << std::endl;
                    for ( size_t s1 = 0; s1 < model.getS(); ++s1 )
                        for ( size_t a = 0; a < model.getA(); ++a )
                            for ( size_t o = 0; o < model.getO(); ++o ) {
                                auto ws1 = WorkerState(s1, n_skills);
                                if (!model.isObservationClamped(s1, a, o)) {
                                    std::cout << "|" << ws1 << "|" << " . " << a << " . " << "|" << o << "|" << " -> " << model.getObservationProbability(s1, a, o);
                                    std::cout << std::endl;
                                }
                            }

                    // Print initial beliefs.
                    std::cout << "---IB---" << std::endl;
                    for ( size_t s = 0; s < model.getS(); ++s ) {
                        auto ws = WorkerState(s, n_skills);
                        if (!model.isInitialBeliefClamped(s)) {
                            std::cout << "|" << ws << "|" << " -> " << model.getInitialBeliefProbability(s);
                            std::cout << std::endl;
                        }
                    }
                    */
                } else {
                    b = b_start;
                }
                // Re-solve always.
                std::unique_ptr<POMDP::Policy> p;
                if (ptype == "ip") {
                    POMDP::IncrementalPruning solver(h, 0.0 /* BUG */);
                    auto solution = solver(model);
                    if (!std::get<0>(solution)) {
                        std::cout << "Solver failed." << std::endl;
                        return 1;
                    }
                    auto vf = std::get<1>(solution);
                    std::unique_ptr<POMDP::Policy> p1(new POMDP::Policy(model.getS(), model.getA(), model.getO(), vf));
                    p = std::move(p1);
                } else if (ptype == "pbvi") {
                    POMDP::PBVI solver(PBVI_NBELIEFS, h, PBVI_EPSILON);
                    auto solution = solver(model);
                    if (!std::get<0>(solution)) {
                        std::cout << "Solver failed." << std::endl;
                        return 1;
                    }
                    auto vf = std::get<1>(solution);
                    std::unique_ptr<POMDP::Policy> p1(new POMDP::Policy(model.getS(), model.getA(), model.getO(), vf));
                    p = std::move(p1);
                }
                //std::cout << "done solving\n";

                time_t rawtime;
                time(&rawtime);
                // Write to file.
                output << it << ","
                       << ep << ","
                       << t << ","
                       << pname << ","
                       //<< ctime(&rawtime) << ","
                       << rawtime << ","
                       << "," // empty action
                       << s << "," 
                       << "," // empty observation
                       << ","; // empty reward
                std::stringstream ss;
                std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
                output << ss.str() << "\n";
                ss.str("");  // clear

                // Run episode.
                while (s != WorkerState::TERM) {
                    std::cout << "t: " << t << std::endl;
                    size_t a;
                    // Choose random valid action (other than booting).
                    if (epsilon && uniformDistribution(rand) <= epsilon.get()) {
                        size_t sample_s = sampleProbability(S, b, rand);
                        auto st = WorkerState(sample_s, n_skills);
                        size_t A = model.getA();
                        std::vector<double> aprobs;
                        for ( size_t a1 = 0; a1 < A; ++a1 ) {
                            if (st.is_valid_action(a1) && a1 != A_BOOT) {
                                aprobs.push_back(1.0);
                            } else {
                                aprobs.push_back(0.0);
                            }
                        }
                        normalizeProbability(aprobs.begin(), aprobs.end(), aprobs.begin());
                        a = sampleProbability(A, aprobs, rand);
                    } else if (ptype == "train") {
                        if (POMDP::beliefExpectedReward(model, b, A_ASK) > 0) {
                            a = A_ASK;
                        } else if (!is_quiz(b, n_skills, S)) {
                            // BUG: Teach only first rule.
                            a = N_RES_A;
                        } else {
                            a = A_EXP;
                        }
                    } else if (ptype == "ip" || ptype == "pbvi") {
                        a = std::get<0>(p->sampleAction(b, h));
                    } else {
                        throw std::invalid_argument( "Unknown policy type" );
                    }
                    auto res = model_true.sampleSOR(s, a);
                    s = std::get<0>(res);
                    size_t o = std::get<1>(res);
                    double r = std::get<2>(res);
                    exp.record(a, o, r);
                    //std::cout << exp.getEpisodeN() << " (" << ep << ") x " << exp.getEventN(it) << " (" << t << ")" << std::endl;
                    //std::cout << a << " " << exp.getAction(ep, t) << std::endl;
                    //std::cout << o << " " << exp.getObservation(ep, t) << std::endl;
                    //std::cout << r << " " << exp.getReward(ep, t) << std::endl << std::endl;
                    //auto st = WorkerState(s, n_skills);
                    //std::cout << "'" << st << "'" << "," << a << "," << o << "," << r << "\n";

                    POMDP::Belief b_new = POMDP::updateBelief(model, b, a, o);
                    b = b_new;  // copies vector

                    time(&rawtime);
                    // Write to file.
                    output << it << ","
                           << ep << ","
                           << ++t << ","
                           << pname << ","
                           //<< ctime(&rawtime) << ","
                           << rawtime << ","
                           << a << ","
                           << s << "," 
                           << o << ","
                           << r << ",";
                    std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
                    output << ss.str() << "\n";
                    ss.str("");
                }
            }  // episode
        }  // policy.
    }  // iteration

    return 0;
}
