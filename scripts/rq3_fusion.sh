#!/usr/bin/env bash
# RQ3 — Operation Fusion Impact
# Compares unfused vs fused (--affine-loop-fusion) for the matmul+relu+bias chain.
# Output: results/rq3_fusion.csv
#
# CSV columns: variant, time_mean_s, time_stddev_s
# Run from project root: bash scripts/rq3_fusion.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results /tmp/mlir_rq3

OUT=results/rq3_fusion.csv
echo "variant,time_mean_s,time_stddev_s" > "$OUT"

WARMUP=3
RUNS=10

NITER=5  # matches hardcoded value in kernels/elementwise/chain.mlir

_bench() {
  local label="$1" lowered="$2"
  local binary=/tmp/mlir_rq3/bin_${label}
  local hfcsv=/tmp/mlir_rq3/hf_${label}.csv

  bash scripts/_compile_native.sh "$lowered" "$binary"
  "$HYPERFINE" \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-csv "$hfcsv" \
    "$binary" \
    2>/dev/null

  MEAN=$(awk   -F',' -v n="$NITER" 'NR==2{printf "%.10f", $2/n}' "$hfcsv")
  STDDEV=$(awk -F',' -v n="$NITER" 'NR==2{printf "%.10f", $3/n}' "$hfcsv")
  echo "$label,$MEAN,$STDDEV" | tee -a "$OUT"
}

# Unfused — separate affine loop nests
bash pipelines/to_affine.sh kernels/elementwise/chain.mlir \
  > /tmp/mlir_rq3/chain_unfused.mlir
_bench "unfused" /tmp/mlir_rq3/chain_unfused.mlir

# Fused — affine-loop-fusion merges adjacent compatible loop nests
bash pipelines/to_affine_fused.sh kernels/elementwise/chain.mlir \
  > /tmp/mlir_rq3/chain_fused.mlir
_bench "fused" /tmp/mlir_rq3/chain_fused.mlir

echo ""
echo "RQ3 results saved to $OUT"
