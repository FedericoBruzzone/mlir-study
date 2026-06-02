#!/usr/bin/env bash
# RQ4 — Workload Breadth
# Benchmarks conv2d and batch_matmul under the best path found in RQ2
# (affine + tiled T=16 for matmul; same strategy applied here).
# Output: results/rq4_workloads.csv
#
# CSV columns: kernel, variant, time_mean_s, time_stddev_s
# Run from project root: bash scripts/rq4_workloads.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results /tmp/mlir_rq4

OUT=results/rq4_workloads.csv
echo "kernel,variant,time_mean_s,time_stddev_s" > "$OUT"

WARMUP=3
RUNS=10

_bench() {
  local kernel="$1" label="$2" lowered="$3" niter="$4"
  local binary=/tmp/mlir_rq4/bin_${kernel}_${label}
  local hfcsv=/tmp/mlir_rq4/hf_${kernel}_${label}.csv

  bash scripts/_compile_native.sh "$lowered" "$binary"
  "$HYPERFINE" \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-csv "$hfcsv" \
    "$binary" \
    2>/dev/null

  MEAN=$(awk   -F',' -v n="$niter" 'NR==2{printf "%.10f", $2/n}' "$hfcsv")
  STDDEV=$(awk -F',' -v n="$niter" 'NR==2{printf "%.10f", $3/n}' "$hfcsv")
  echo "$kernel,$label,$MEAN,$STDDEV" | tee -a "$OUT"
}

# ── Conv2d  56×56 (ResNet-like feature map) ────────────────────────────────
NITER_CONV=10
CONV=/tmp/mlir_rq4/conv2d_56.mlir
sed -e 's/OSIZE/56/g' -e 's/SIZE/58/g' -e "s/NITER/$NITER_CONV/g" \
  kernels/conv2d/bench.mlir.tpl > "$CONV"

# Path A — untiled affine
bash pipelines/to_affine.sh "$CONV" > /tmp/mlir_rq4/conv_affine.mlir
_bench conv2d affine /tmp/mlir_rq4/conv_affine.mlir "$NITER_CONV"

# Path C — tiled affine (T=16, best from RQ1 for matmul; heuristic for conv)
bash pipelines/to_affine_tiled.sh "$CONV" 16 > /tmp/mlir_rq4/conv_tiled.mlir
_bench conv2d affine_tiled_16 /tmp/mlir_rq4/conv_tiled.mlir "$NITER_CONV"

# ── Batch matmul  16×128×128  (attention-scale heads) ──────────────────────
NITER_BGEMM=20
BGEMM=/tmp/mlir_rq4/bgemm_16x128.mlir
sed -e 's/BATCH/16/g' -e 's/SIZE/128/g' -e "s/NITER/$NITER_BGEMM/g" \
  kernels/batch_matmul/bench.mlir.tpl > "$BGEMM"

bash pipelines/to_affine.sh "$BGEMM" > /tmp/mlir_rq4/bgemm_affine.mlir
_bench batch_matmul affine /tmp/mlir_rq4/bgemm_affine.mlir "$NITER_BGEMM"

bash pipelines/to_affine_tiled.sh "$BGEMM" 16 > /tmp/mlir_rq4/bgemm_tiled.mlir
_bench batch_matmul affine_tiled_16 /tmp/mlir_rq4/bgemm_tiled.mlir "$NITER_BGEMM"

# ── Softmax 512×512 (memory-bandwidth dominated) ───────────────────────────
# NITER=20 is hardcoded in kernels/reduction/softmax.mlir
bash pipelines/to_scf.sh kernels/reduction/softmax.mlir > /tmp/mlir_rq4/softmax.mlir
_bench softmax scf /tmp/mlir_rq4/softmax.mlir 20

echo ""
echo "RQ4 results saved to $OUT"
