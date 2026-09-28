SUBDIRS := bfs \
	bias_add \
	embed_scale \
	flash_attention \
	flash_attention_mx \
	flash_attention_mx_fp6 \
	flash_attention_mx_gemma \
	flash_attention_mx_gqa \
	flash_attention_mx_stable \
	fmastore \
	gaussian \
	geglu \
	gemma_4norm \
	gemm_mxgemmini \
	gemm_mxgemmini_ws_restream \
	gemm_simt \
	gemv \
	launch \
	layernorm \
	logit_softcap \
	nearn \
	patch_embed \
	pointerchase \
	qkv_bias \
	rmsnorm \
	rmsnorm_gemma \
	rmsnorm_qkv_fused \
	rope_qkv_fused \
	saxpy \
	sfilter \
	simt_quant \
	softmax \
	special_fn \
	spin \
	swiglu \
	test_backing_L2_fill \
	test_backing_mem \
	ubench \
	vecadd

.PHONY: all $(SUBDIRS) clean clean-all data

# Default target: run make in all subdirectories
all: $(SUBDIRS)

# Deterministically regenerate every kernel's committed data (see regen_data.sh).
# Needs python3 with numpy (+ torch for the flash/fused generators) and the
# lib/mxgemmini submodule checked out. Override the interpreter with PYTHON=...
data:
	PYTHON=$(if $(PYTHON),$(PYTHON),python3) ./regen_data.sh

$(SUBDIRS):
	$(MAKE) -C $@

# Clean target: run make clean in all subdirectories
clean:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done

clean-all:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean-all; done
