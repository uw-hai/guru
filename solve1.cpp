/* BUG: Description */
#include <iostream>
#include <iomanip>  // for string padding
#include <fstream>
#include <random>

#include <boost/program_options.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

/*
#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>
#include <AIToolbox/POMDP/Types.hpp>
*/

#include "WorkLearnProblem.hpp"

const int SEED = 0;


/* Parameters */
// Set arbitrary cost for now. This is really lambda * actual cost, to
// trade off between value of additional correct answers and cost.
/*
//constexpr double cost = -10;
constexpr double COST = -.1;
//constexpr double cost_living = 0.0;
constexpr double COST_LIVING = -.001;
double P_LEARN = 0.4;
double P_LEAVE = 0.01;
double P_SLIP = 0.1;
double P_GUESS = 0.5;
double P_R = 0.5;
//double P_R = 0.0;

//double P_1 = 0.4;
double P_1 = 0.5;
*/


namespace po = boost::program_options;
using namespace AIToolbox;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("cost", po::value<double>(), "cost parameter")
        ("cost_living", po::value<double>(), "cost_living parameter")
        ("p_learn", po::value<double>(), "p_learn parameter")
        ("p_leave", po::value<double>(), "p_leave parameter")
        ("p_slip", po::value<double>(), "p_slip parameter")
        ("p_guess", po::value<double>(), "p_guess parameter")
        ("p_r", po::value<double>(), "p_r parameter")
        ("p_s0", po::value<double>(), "prior probability worker has skill 0")
        ("p_1", po::value<double>(), "p_1 parameter")
        ("max_horizon", po::value<unsigned>(), "maximum horizon")
        ("horizon", po::value< std::vector<unsigned> >(), "policy horizon")
        ("iterations", po::value<unsigned>(), "number of iterations")
        ("discount", po::value<double>(), "discount parameter")
        ("exp_name", po::value<std::string>(), "experiment name")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 

    const double cost = vm["cost"].as<double>();
    const double cost_living = vm["cost_living"].as<double>();
    const double p_learn = vm["p_learn"].as<double>();
    const double p_leave = vm["p_leave"].as<double>();
    const double p_slip = vm["p_slip"].as<double>();
    const double p_guess = vm["p_guess"].as<double>();
    const double p_r = vm["p_r"].as<double>();
    const double p_s0 = vm["p_s0"].as<double>();
    const double p_1 = vm["p_1"].as<double>();
    const unsigned max_horizon = vm["max_horizon"].as<unsigned>();
    const std::vector<unsigned> horizon = vm["horizon"].as< std::vector<unsigned> >();
    const unsigned iterations = vm["iterations"].as<unsigned>();
    const std::string exp_name = vm["exp_name"].as<std::string>();
    const double discount = vm["discount"].as<double>();

    std::cout << "Loading...\n";
    auto model = makeWorkLearnProblem(cost, cost_living, p_learn, p_leave, p_slip, p_guess, p_r, p_1);
    model.setDiscount(discount);

    // Ground truth proxy.
    std::cout << "Solving...\n";
    POMDP::IncrementalPruning groundTruth(max_horizon, 0.0);
    auto solution = groundTruth(model);
    auto& vf = std::get<1>(solution);
    POMDP::Policy p(model.getS(), model.getA(), model.getO(), vf);

    if (!std::get<0>(solution)) {
        std::cout << "Solver failed." << std::endl;
        return 1;
    }

    std::cout << "Saving policy to file...\n";
    {
        std::ofstream output("res/" + exp_name + "_p.txt");
        output << p;
    }

    // Sample actions from various states.
    /*
    std::vector<POMDP::Belief> beliefs{
        {1.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0}};
    std::vector<POMDP::Belief> beliefs{
        {0.1, 0.9, 0.0, 0.0, 0.0},
        {0.2, 0.8, 0.0, 0.0, 0.0},
        {0.25, 0.75, 0.0, 0.0, 0.0},
        {0.3, 0.7, 0.0, 0.0, 0.0},
        {0.4, 0.6, 0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0, 0.0, 0.0},
        {0.6, 0.4, 0.0, 0.0, 0.0},
        {0.7, 0.3, 0.0, 0.0, 0.0},
        {0.8, 0.2, 0.0, 0.0, 0.0},
        {0.9, 0.1, 0.0, 0.0, 0.0}};
    for ( auto & b : beliefs ) {
        auto a = p.sampleAction(b);
        //std::cout << b << std::endl;
        std::cout << a << std::endl;
    }
    */

    //////////////////////////////////
    //std::default_random_engine rand_(Impl::Seeder::getSeed());
    std::default_random_engine rand_(SEED);
    static std::uniform_real_distribution<double> uniformDistribution(0.0, 1.0);

    std::ofstream output("res/" + exp_name + ".txt");
    output << "iteration,t,policy,a,s,o,r,b\n";

    POMDP::Belief b_start = {1.0 - p_s0, p_s0, 0.0, 0.0, 0.0};
    POMDP::Belief b_term = {0.0, 0.0, 0.0, 0.0, 1.0};
    for ( unsigned it = 0; it < iterations; ++it ) {
        std::cout << "Iteration " << it << std::endl;

        size_t s_start;
        if (uniformDistribution(rand_) <= p_s0) {
            s_start = 1;
        } else {
            s_start = 0;
        }

        // Iterate over policies.
        for (const unsigned h : horizon) {

            POMDP::Belief b = b_start;
            size_t s = s_start;
            unsigned t = 0;
            output << it << ","
                   << t << ","
                   << "h" << std::setfill('0') << std::setw(2) << h << "," // horizon policy
                   << "," // empty action
                   << s << "," 
                   << "," // empty observation
                   << ","; // empty reward
            std::stringstream ss;
            std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
            output << ss.str() << "\n";
            ss.str("");  // clear.

            while (b != b_term) {
                t++;
                size_t a = std::get<0>(p.sampleAction(b, h));
                auto res = model.sampleSOR(s, a);
                s = std::get<0>(res);
                size_t o = std::get<1>(res);
                double r = std::get<2>(res);

                POMDP::Belief b_new = POMDP::updateBelief(model, b, a, o);
                b = b_new;  // Copies vector.

                output << it << ","
                       << t << ","
                       << "h" << std::setfill('0') << std::setw(2) << h << "," // horizon policy
                       << a << ","
                       << s << "," 
                       << o << ","
                       << r << ",";
                std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
                output << ss.str() << "\n";
                ss.str("");
            }
        }  // end policy run

        // Special training policy.
        // BUG: Duplicate code.
        POMDP::Belief b = b_start;
        size_t s = s_start;
        unsigned t = 0;
        output << it << ","
               << t << ","
               << "train,"
               << "," // empty action
               << s << "," 
               << "," // empty observation
               << ","; // empty reward
        std::stringstream ss;
        std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
        output << ss.str() << "\n";
        ss.str("");  // clear.

        while (b != b_term) {
            t++;
            size_t a;
            if (POMDP::beliefExpectedReward(model, b, A_ASK) > 0) {
                a = A_ASK;
            } else if (b[0] > 0) /* Non-quiz state */ {
                a = A_QUIZ_0;
            } else {
                a = A_EXP_0;
            }
            auto res = model.sampleSOR(s, a);
            s = std::get<0>(res);
            size_t o = std::get<1>(res);
            double r = std::get<2>(res);

            POMDP::Belief b_new = POMDP::updateBelief(model, b, a, o);
            b = b_new;  // Copies vector.

            output << it << ","
                   << t << ","
                   << "train,"
                   << a << ","
                   << s << "," 
                   << o << ","
                   << r << ",";
            std::copy(b.begin(), b.end(), std::ostream_iterator<double>(ss, " "));
            output << ss.str() << "\n";
            ss.str("");
        }


    }  // end iteration

    return 0;
}
