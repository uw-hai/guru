#define BOOST_TEST_MODULE UTIL
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <math.h>       /* exp */ 
#include <vector>
#include <WorkLearn/util.hpp>

BOOST_AUTO_TEST_CASE( one ) {
    std::vector<double> v {log(0.1)};
    double d = logsumexp(v);
    BOOST_CHECK_EQUAL(d, log(0.1));
}

BOOST_AUTO_TEST_CASE( two ) {
    std::vector<double> v {log(0.1), log(0.2)};
    double d = logsumexp(v);
    BOOST_CHECK_EQUAL(d, log(0.1 + 0.2));
}
