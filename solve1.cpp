/* BUG: Description */
#include <iostream>
#include <iomanip>  // for string padding
#include <fstream>
#include <random>
#include <map>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
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
    const size_t iterations = pt.get<size_t>("params.iterations");
    const double discount = pt.get<double>("params.discount");
    ptree &policies = pt.get_child("policies");

    // Process input.
    std::cout << "Loading...\n";
    size_t n_skills = p_r.size();
    size_t S = (n_skills + 1) * std::pow(2, n_skills) + 1;
    auto model = makeWorkLearnProblem(cost, cost_exp, cost_living, p_learn, p_leave, p_slip, p_guess, p_r, p_1, n_skills, S);
    model.setDiscount(discount);

    // Ground truth proxy.
    std::cout << "Solving...\n";

    // Find maximum horizons.
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

    // BUG: Fix duplicate code.
    std::map<std::string, POMDP::Policy*> policy_map;
    if (max_horizon_ip > 0) {
        POMDP::IncrementalPruning solver(max_horizon_ip, 0.0 /* BUG */);
        auto solution = solver(model);
        if (!std::get<0>(solution)) {
            std::cout << "Solver failed." << std::endl;
            return 1;
        }
        auto& vf = std::get<1>(solution);
        auto* p = new POMDP::Policy(model.getS(), model.getA(), model.getO(), vf);
        policy_map["ip"] = p;
        std::cout << "Saving policy to file...\n";
        {
            std::ofstream output("res/" + exp_name + "_p_ip.txt");
            output << p;
        }
    }
    if (max_horizon_pbvi > 0) {
        POMDP::PBVI solver(PBVI_NBELIEFS, max_horizon_pbvi, PBVI_EPSILON);
        auto solution = solver(model);
        if (!std::get<0>(solution)) {
            std::cout << "Solver failed." << std::endl;
            return 1;
        }
        auto& vf = std::get<1>(solution);
        auto* p = new POMDP::Policy(model.getS(), model.getA(), model.getO(), vf);
        policy_map["pbvi"] = p;
        std::cout << "Saving policy to file...\n";
        {
            std::ofstream output("res/" + exp_name + "_p_pbvi.txt");
            output << p;
        }
    }

    // Perform experiments
    // Make experiments reproducible. For true randomness, use:
    //      std::default_random_engine rand(Impl::Seeder::getSeed());
    std::default_random_engine rand(SEED);
    static std::uniform_real_distribution<double> uniformDistribution(0.0, 1.0);

    std::ofstream output("res/" + exp_name + ".txt");
    output << "iteration,t,policy,a,s,o,r,b\n";

    POMDP::Belief b_start;
    POMDP::Belief b_term;
    for ( size_t s = 0; s < S; ++s ) {
        auto st = WorkerState(s, n_skills);
        if (st.is_term()) {
            b_term.push_back(1.0);
        } else {
            b_term.push_back(0.0);
        }

        if (st.is_term() || st.quiz_val() != 0) {
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
    for ( size_t it = 0; it < iterations; ++it ) {
        std::cout << "Iteration " << it << std::endl;
        size_t s_start = sampleProbability(S, b_start, rand);

        // Iterate over policies.
        for (auto& pdef : policies) {
            std::string pname = pdef.second.get<std::string>("name");
            std::string ptype = pdef.second.get<std::string>("type");
            size_t h;
            if (ptype == "ip" || ptype == "pbvi") {
                h = pdef.second.get<size_t>("horizon");
            }

            POMDP::Belief b = b_start;
            size_t s = s_start;
            size_t t = 0;
            output << it << ","
                   << t << ","
                   << pname << ","
                   << "," // empty action
                   << s << "," 
                   << "," // empty observation
                   << ","; // empty reward
            std::stringstream ss;
            std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
            output << ss.str() << "\n";
            ss.str("");  // clear

            while (b != b_term) {
                size_t a;
                if (ptype == "train") {
                    if (POMDP::beliefExpectedReward(model, b, A_ASK) > 0) {
                        a = A_ASK;
                    } else if (!is_quiz(b, n_skills, S)) {
                        // BUG: Teach only first rule.
                        a = N_RES_A;
                    } else {
                        a = A_EXP;
                    }
                } else if (ptype == "ip") {
                    a = std::get<0>(policy_map["ip"]->sampleAction(b, h));
                } else if (ptype == "pbvi") {
                    a = std::get<0>(policy_map["pbvi"]->sampleAction(b, h));
                } else {
                    throw std::invalid_argument( "Unknown policy type" );
                }
                auto res = model.sampleSOR(s, a);
                s = std::get<0>(res);
                size_t o = std::get<1>(res);
                double r = std::get<2>(res);
                ++t;

                POMDP::Belief b_new = POMDP::updateBelief(model, b, a, o);
                b = b_new;  // copies vector

                output << it << ","
                       << t << ","
                       << pname << ","
                       << a << ","
                       << s << "," 
                       << o << ","
                       << r << ",";
                std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
                output << ss.str() << "\n";
                ss.str("");
            }
        }  // end policy run

    }  // end iteration

    // Save state names.
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


    return 0;
}
