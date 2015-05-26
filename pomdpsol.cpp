/* BUG: Description */
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/POMDP/IO.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>

namespace po = boost::program_options;
using namespace AIToolbox;

int main(int argc, char **argv) {
    // Parse input.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("input,i", po::value<std::string>()->required(), "POMDP input file path")
        ("output,o", po::value<std::string>()->required(), "Policy output file path")
        ("discount,d", po::value<double>()->required(), "Discount factor")
        ("horizon,h", po::value<size_t>()->required(), "Maximum horizon")
        ("n_observations,no", po::value<size_t>()->required(), "Number of observations")
        ("n_states,ns", po::value<size_t>()->required(), "Number of states")
        ("n_actions,na", po::value<size_t>()->required(), "Number of actions")
        ("pbvi_epsilon,", po::value<double>()->default_value(0.01), "")
        ("pbvi_nbeliefs,", po::value<size_t>()->default_value(1000), "Maximum number of beliefs for PBVI")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 
    std::string inputPath = vm["input"].as<std::string>();
    std::string outputPath = vm["output"].as<std::string>();
    double discount = vm["discount"].as<double>();
    size_t horizon = vm["horizon"].as<size_t>();
    size_t S = vm["n_states"].as<size_t>();
    size_t O = vm["n_observations"].as<size_t>();
    size_t A = vm["n_actions"].as<size_t>();
    double pbvi_epsilon = vm["pbvi_epsilon"].as<double>();
    size_t pbvi_nbeliefs = vm["pbvi_nbeliefs"].as<size_t>();
    POMDP::Model<MDP::Model> m(O, S, A);
    {
        std::ifstream inputFile(inputPath);
        POMDP::operator>>(inputFile, m);
    }
    m.setDiscount(discount);

    POMDP::PBVI solver(pbvi_nbeliefs, horizon, pbvi_epsilon);
    auto solution = solver(m);
    if (!std::get<0>(solution)) {
        std::cout << "Solver failed." << std::endl;
        return 1;
    }
    auto vf = std::get<1>(solution);
    POMDP::Policy pol(m.getS(), m.getA(), m.getO(), vf);
    {
        std::ofstream output(outputPath);
        output << pol;
    }

    return 0;
}
