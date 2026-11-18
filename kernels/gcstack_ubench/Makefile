PROJECT = gcstack_ubench

# Every bench_*.cpp is its own kernel -> bench_<name>.radiance.elf + bench_<name>.soc.elf.
# Add a bench by dropping in a new bench_<name>.cpp (start from _template.cpp).
MU_SRCS = $(sort $(wildcard bench_*.cpp))
HOST_SRCS = host.cpp

# Global knobs, applied to every bench: make NUM_WARPS=1 ITERS=16384
ifneq ($(strip $(NUM_WARPS)),)
MU_CFLAGS += -DUBENCH_NUM_WARPS=$(NUM_WARPS)
endif
ifneq ($(strip $(ITERS)),)
MU_CFLAGS += -DUBENCH_ITERS=$(ITERS)
endif

include ../common.mk

$(MU_SRCS:.cpp=.mu.o): ubench.hpp

.PHONY: list
list:
	@for f in $(MU_SRCS); do echo "$${f%.cpp}.soc.elf"; done

# soc/fuse_rv32_into_rv64.sh writes a fixed ./start.o per link, so parallel
# .soc.elf links clobber each other. The whole build takes seconds serially.
.NOTPARALLEL:
