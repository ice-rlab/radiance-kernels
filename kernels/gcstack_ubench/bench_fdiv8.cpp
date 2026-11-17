// bench_fdiv8 -- bench_fdiv with all 8 warps: does the FP divide unit overlap warps,
// and how does GCStack classify 8 warps waiting on one ~86-cycle divider?
#define UBENCH_NUM_WARPS 8
#define UBENCH_ITERS 2
#include "bench_fdiv.cpp"
