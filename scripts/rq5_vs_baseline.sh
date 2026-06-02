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
  local label="$1" cmd="$2" size="$3" niter="$4"
  local hfcsv=/tmp/mlir_rq5/hf_${size}_${label}.csv

  "$HYPERFINE" \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-csv "$hfcsv" \
    "$cmd" \
    2>/dev/null

  MEAN=$(awk   -F',' -v n="$niter" 'NR==2{printf "%.10f", $2/n}' "$hfcsv")
  STDDEV=$(awk -F',' -v n="$niter" 'NR==2{printf "%.10f", $3/n}' "$hfcsv")
  GF=$(_gflops "$size" "$MEAN")
  echo "$size,$label,$MEAN,$STDDEV,$GF" | tee -a "$OUT"
}

for N in "${SIZES[@]}"; do
  case "$N" in
    128)  NITER=100 ;;
    256)  NITER=20  ;;
    512)  NITER=5   ;;
    1024) NITER=2   ;;
    *)    NITER=5   ;;
  esac

  KERNEL=/tmp/mlir_rq5/matmul_${N}.mlir
  sed -e "s/SIZE/$N/g" -e "s/NITER/$NITER/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

  # ── MLIR best path (affine + tile T=16) ──────────────────────────────────
  LOWERED=/tmp/mlir_rq5/lowered_${N}.mlir
  BINARY=/tmp/mlir_rq5/bin_${N}
  bash pipelines/to_affine_tiled.sh "$KERNEL" 16 > "$LOWERED"
  bash scripts/_compile_native.sh "$LOWERED" "$BINARY"
  _bench "mlir_affine_t16" "$BINARY" "$N" "$NITER"

  # ── MLIR explicit vectorization (tile + affine-super-vectorize) ───────────
  LOWERED_V=/tmp/mlir_rq5/lowered_${N}_vec.mlir
  BINARY_V=/tmp/mlir_rq5/bin_${N}_vec
  if bash pipelines/to_vector.sh "$KERNEL" 16 > "$LOWERED_V" 2>/dev/null && \
     bash scripts/_compile_native.sh "$LOWERED_V" "$BINARY_V" 2>/dev/null; then
    _bench "mlir_vector_t16" "$BINARY_V" "$N" "$NITER"
  else
    echo "$N,mlir_vector_t16,N/A,N/A,N/A" | tee -a "$OUT"
  fi

  # ── MLIR interchange + vectorization (fixes loop order, masked loads remain) ─
  LOWERED_VI=/tmp/mlir_rq5/lowered_${N}_vec_interchange.mlir
  BINARY_VI=/tmp/mlir_rq5/bin_${N}_vec_interchange
  if bash pipelines/to_vector_interchange.sh "$KERNEL" 16 > "$LOWERED_VI" 2>/dev/null && \
     bash scripts/_compile_native.sh "$LOWERED_VI" "$BINARY_VI" 2>/dev/null; then
    _bench "mlir_vector_interchange_t16" "$BINARY_VI" "$N" "$NITER"
  else
    echo "$N,mlir_vector_interchange_t16,N/A,N/A,N/A" | tee -a "$OUT"
  fi

  # ── Apple Accelerate baseline ─────────────────────────────────────────────
  BLAS_BIN=baselines/blas_matmul_${N}
  if [[ -x "$BLAS_BIN" ]]; then
    _bench "accelerate_blas" "./$BLAS_BIN $NITER" "$N" "$NITER"
  else
    echo "$N,accelerate_blas,NOT_BUILT,N/A,N/A" | tee -a "$OUT"
  fi
done

echo ""
echo "RQ5 results saved to $OUT"
