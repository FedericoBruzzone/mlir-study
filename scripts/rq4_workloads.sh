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
  local kernel="$1" label="$2" lowered="$3"
  local hfcsv=/tmp/mlir_rq4/hf_${kernel}_${label}.csv

  "$HYPERFINE" \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-csv "$hfcsv" \
    "bash scripts/_run_mlir.sh $lowered" \
    2>/dev/null

  MEAN=$(awk   -F',' 'NR==2{print $2}' "$hfcsv")
  STDDEV=$(awk -F',' 'NR==2{print $3}' "$hfcsv")
  echo "$kernel,$label,$MEAN,$STDDEV" | tee -a "$OUT"
}

# ── Conv2d  56×56 (ResNet-like feature map) ────────────────────────────────
CONV=/tmp/mlir_rq4/conv2d_56.mlir
sed -e 's/OSIZE/56/g' -e 's/SIZE/58/g' kernels/conv2d/bench.mlir.tpl > "$CONV"

# Path A — untiled affine
bash pipelines/to_affine.sh "$CONV" > /tmp/mlir_rq4/conv_affine.mlir
_bench conv2d affine /tmp/mlir_rq4/conv_affine.mlir

# Path C — tiled affine (T=16, best from RQ1 for matmul; heuristic for conv)
bash pipelines/to_affine_tiled.sh "$CONV" 16 > /tmp/mlir_rq4/conv_tiled.mlir
_bench conv2d affine_tiled_16 /tmp/mlir_rq4/conv_tiled.mlir

# ── Batch matmul  16×128×128  (attention-scale heads) ──────────────────────
BGEMM=/tmp/mlir_rq4/bgemm_16x128.mlir
sed -e 's/BATCH/16/g' -e 's/SIZE/128/g' kernels/batch_matmul/bench.mlir.tpl > "$BGEMM"

bash pipelines/to_affine.sh "$BGEMM" > /tmp/mlir_rq4/bgemm_affine.mlir
_bench batch_matmul affine /tmp/mlir_rq4/bgemm_affine.mlir

bash pipelines/to_affine_tiled.sh "$BGEMM" 16 > /tmp/mlir_rq4/bgemm_tiled.mlir
_bench batch_matmul affine_tiled_16 /tmp/mlir_rq4/bgemm_tiled.mlir

# ── Softmax 512×512 (memory-bandwidth dominated) ───────────────────────────
bash pipelines/to_scf.sh kernels/reduction/softmax.mlir > /tmp/mlir_rq4/softmax.mlir
_bench softmax scf /tmp/mlir_rq4/softmax.mlir

echo ""
echo "RQ4 results saved to $OUT"
