PROJECT = kernel

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(MU_OBJS): tf_gemm.hpp transformer_mx_impl.hpp tf_beacon.h include/tf_data.h
host.host.o: tf_beacon.h $(wildcard include/tf_golden.h)

include ../common.mk

# Build the device kernel at -Os, NOT common.mk's -O3.
#
# lib/include/mu_intrinsics.h documents two workarounds for mu_barrier() being duplicated into both
# paths of a warp-divergent branch (it then executes twice under SIMT serialization and deadlocks):
# an explicit `else { nop }` on every tid-guarded block before a barrier, and -Os, which keeps the
# compiler from branch-duplicating. kernel.cpp does the first everywhere; this is the second.
# The previous version of this kernel hung at a barrier with Gemmini provably idle, and moving code
# AFTER the barrier changed whether it passed -- a codegen symptom, not a logic bug.
#
# MUST come after the include: common.mk does `MU_CFLAGS += -O3`, and make expands MU_CFLAGS at
# recipe time, so appending here puts -Os last on the command line, where it wins.
#
# Schedule-perturbation builds: make clean && make EXTRA_MU_CFLAGS=-DTF_PHASE2  (see tf_gemm.hpp)
MU_CFLAGS += -Os

# Device addresses of the per-phase DRAM activation buffers, read out of the LINKED GPU image so
# host.cpp can dump them after the run without the GPU code changing at all. Regenerated whenever
# the GPU elf is relinked; host.host.o depends on it, so the fused soc.elf always matches.
include/tf_syms.h: kernel.radiance.elf
	$(LLVM_MUON)/bin/llvm-nm --defined-only $< | \
	  awk '$$3 ~ /^(_ZL[0-9]+)?g_(q|k|v|s|s0|ctx|res|ff|out|af8|afs|q0f8|q0fs|ktf8|kts|sraw|q0|k0|kt|v0|asmem|bsmem)$$/ { n = $$3; sub(/^_ZL[0-9]+/, "", n); \
	       printf "#define TFSYM_%s 0x%su\n", n, $$1 }' > $@
	@# Early TF_UPTO rungs never touch the later buffers, so the linker drops them: define the missing
	@# ones as 0 (the host only reads buffers of rungs that ran), but g_q must always exist.
	@grep -q TFSYM_g_q $@ || { echo "tf_syms.h: g_q missing"; cat $@; exit 1; }
	@for s in g_q g_k g_v g_s g_ctx g_res g_ff g_af8 g_afs g_q0f8 g_q0fs g_ktf8 g_kts g_sraw g_q0 g_k0 g_kt g_asmem g_bsmem; do \
	  grep -q "TFSYM_$$s " $@ || printf '#define TFSYM_%s 0u\n' $$s >> $@; done
host.host.o: include/tf_syms.h
