// bench_empty -- baseline: launch + teardown only, no body.
//
// The gcstack window covers the whole GPU run, including mu_schedule's warp
// spawn and the exit. Run this alongside the others and subtract its bucket
// cycles (not fractions) to get each bench's steady-state stack.

#include "ubench.hpp"

UBENCH_BODY(u) {
  ubench_sink(u, u.tid);
}
