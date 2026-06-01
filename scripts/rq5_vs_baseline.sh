#!/usr/bin/env bash
# RQ5 — MLIR vs Apple Accelerate Baseline
# Compares the best MLIR path for each size against cblas_sgemm (Accelerate).
# Output: results/rq5_vs_baseline.csv
#
# CSV columns: size_N, variant, time_mean_s, time_stddev_s, gflops
# Run from project root: bash scripts/rq5_vs_baseline.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results /tmp/mlir_rq5

OUT=results/rq5_vs_baseline.csv
echo "size_N,variant,time_mean_s,time_stddev_s,gflops" > "$OUT"

SIZES=(128 256 512 1024)
WARMUP=5
RUNS=20   # more runs for publication-quality statistics

_gflops() {
  local n="$1" t="$2"
  awk "BEGIN{printf \"%.2f\", (2.0*$n*$n*$n) / ($t * 1e9)}"
}

_bench() {
  local label="$1" cmd="$2" size="$3"
  local hfcsv=/tmp/mlir_rq5/hf_${size}_${label}.csv

  "$HYPERFINE" \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-csv "$hfcsv" \
    "$cmd" \
    2>/dev/null

  MEAN=$(awk   -F',' 'NR==2{print $2}' "$hfcsv")
  STDDEV=$(awk -F',' 'NR==2{print $3}' "$hfcsv")
  GF=$(_gflops "$size" "$MEAN")
  echo "$size,$label,$MEAN,$STDDEV,$GF" | tee -a "$OUT"
}

for N in "${SIZES[@]}"; do
  # ── MLIR best path (affine + tile T=16) ──────────────────────────────────
  KERNEL=/tmp/mlir_rq5/matmul_${N}.mlir
  LOWERED=/tmp/mlir_rq5/lowered_${N}.mlir
  sed "s/SIZE/$N/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"
  bash pipelines/to_affine_tiled.sh "$KERNEL" 16 > "$LOWERED"
  _bench "mlir_affine_t16" "bash scripts/_run_mlir.sh $LOWERED" "$N"

  # ── MLIR vectorized (tile + NEON) ─────────────────────────────────────────
  LOWERED_V=/tmp/mlir_rq5/lowered_${N}_vec.mlir
  if bash pipelines/to_vector.sh "$KERNEL" 16 > "$LOWERED_V" 2>/dev/null; then
    _bench "mlir_vector_t16" "bash scripts/_run_mlir.sh $LOWERED_V" "$N"
  else
    echo "$N,mlir_vector_t16,N/A,N/A,N/A" | tee -a "$OUT"
  fi

  # ── Apple Accelerate baseline ─────────────────────────────────────────────
  BLAS_BIN=baselines/blas_matmul_${N}
  if [[ -x "$BLAS_BIN" ]]; then
    _bench "accelerate_blas" "./$BLAS_BIN" "$N"
  else
    echo "$N,accelerate_blas,NOT_BUILT,N/A,N/A" | tee -a "$OUT"
  fi
done

echo ""
echo "RQ5 results saved to $OUT"
