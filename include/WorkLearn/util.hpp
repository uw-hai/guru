#include <stddef.h>  /* size_t */
#include <math.h>       /* exp */ 
#include <vector>

// Sums logarithms in exponentiated space safely.
inline double logsumexp(const std::vector<double>& nums) {
    double max_exp = nums[0], sum = 0.0;

    for (double d : nums)
        if (d > max_exp)
            max_exp = d;

    for (double d : nums)
        sum += exp(d - max_exp);

    return log(sum) + max_exp;
}
