SHELL := /usr/bin/env bash

.PHONY: all verify collect-env baselines models rq1 rq2 rq3 rq4 rq5 rq-iree roofline llvm-mca clean

# Full study: env → verify → baselines → all RQs → analysis → roofline
all: collect-env verify baselines models rq1 rq2 rq3 rq4 rq5 rq-iree llvm-mca roofline

# Record hardware + tool versions (run once before experiments)
collect-env:
	bash scripts/collect_env.sh

# End-to-end correctness check (< 10 s)
verify:
	bash scripts/verify.sh

# Build Apple Accelerate baseline binaries
baselines:
	$(MAKE) -C baselines all

# RQ1 — Tiling Sensitivity (matmul)
rq1:
	mkdir -p results
	bash scripts/rq1_sweep_tiles.sh

# RQ2 — Lowering Path Comparison (matmul: affine / scf / tiled / vectorized)
rq2:
	mkdir -p results
	bash scripts/rq2_compare_paths.sh

# RQ3 — Operation Fusion Impact (matmul + relu + bias)
rq3:
	mkdir -p results
	bash scripts/rq3_fusion.sh

# RQ4 — Workload Breadth (conv2d + batch_matmul, best path from RQ2)
rq4:
	mkdir -p results
	bash scripts/rq4_workloads.sh

# RQ5 — MLIR vs Baseline (BLAS/Accelerate ceiling)
rq5:
	mkdir -p results
	bash scripts/rq5_vs_baseline.sh

# Export real PyTorch models to MLIR via iree-turbine
models:
	.venv/bin/python3 scripts/export_models.py

# Benchmark real models through IREE (production path).
# rq_iree.sh  → rq_iree.csv      (wall-clock, includes module-load overhead)
# bench_iree_runtime.py → rq_iree_clean.csv  (per-call, no load overhead)
# Both must run together: the vmfb files compiled by rq_iree.sh
# live in /tmp/mlir_iree/ and are needed by bench_iree_runtime.py.
rq-iree:
	mkdir -p results /tmp/mlir_iree
	bash scripts/rq_iree.sh
	.venv/bin/python3 scripts/bench_iree_runtime.py

# Static IPC analysis via llvm-mca (requires source-built LLVM 23 in llvm-project/build)
llvm-mca:
	mkdir -p results
	bash scripts/llvm_mca_analysis.sh

# Roofline model analysis: text table + PDF/PNG plot
roofline:
	mkdir -p results
	bash scripts/roofline.sh
	.venv/bin/python3 scripts/plot_roofline.py

# Remove generated artifacts (results are NOT deleted)
clean:
	rm -rf /tmp/mlir_rq* /tmp/mlir_smoke_* /tmp/mlir_aot_*
	$(MAKE) -C baselines clean
