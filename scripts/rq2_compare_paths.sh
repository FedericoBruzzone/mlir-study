#!/usr/bin/env bash
# RQ2 — Lowering Path Comparison
# Compares Path A (affine), Path B (scf), Path C (affine+tiled) for matmul sizes.
# Output: results/rq2_paths.csv
#
# CSV columns: size_N, path, tile_T, time_mean_s, time_stddev_s
# Run from project root: bash scripts/rq2_compare_paths.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results /tmp/mlir_rq2

OUT=results/rq2_paths.csv
echo "size_N,path,tile_T,time_mean_s,time_stddev_s" > "$OUT"

SIZES=(128 256 512 1024)
DEFAULT_TILE=16   # T=16 is empirically optimal on M4 (see RQ1); T=64 causes cache aliasing
WARMUP=3
RUNS=10

_bench() {
  local label="$1" lowered="$2" size="$3" tile="$4"
  local hfcsv=/tmp/mlir_rq2/hf_${size}_${label}.csv

  "$HYPERFINE" \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-csv "$hfcsv" \
    "bash scripts/_run_mlir.sh $lowered" \
    2>/dev/null

  MEAN=$(awk -F',' 'NR==2{print $2}' "$hfcsv")
  STDDEV=$(awk -F',' 'NR==2{print $3}' "$hfcsv")
  echo "$size,$label,$tile,$MEAN,$STDDEV" | tee -a "$OUT"
}

for N in "${SIZES[@]}"; do
  KERNEL=/tmp/mlir_rq2/matmul_${N}.mlir
  sed "s/SIZE/$N/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

  # Path A — affine (no tiling)
  bash pipelines/to_affine.sh "$KERNEL" > /tmp/mlir_rq2/affine_${N}.mlir
  _bench "affine" /tmp/mlir_rq2/affine_${N}.mlir "$N" "none"

  # Path B — scf (no tiling)
  bash pipelines/to_scf.sh "$KERNEL" > /tmp/mlir_rq2/scf_${N}.mlir
  _bench "scf" /tmp/mlir_rq2/scf_${N}.mlir "$N" "none"

  # Path C — affine + tiled (DEFAULT_TILE)
  bash pipelines/to_affine_tiled.sh "$KERNEL" "$DEFAULT_TILE" \
    > /tmp/mlir_rq2/affine_tiled_${N}.mlir
  _bench "affine_tiled" /tmp/mlir_rq2/affine_tiled_${N}.mlir "$N" "$DEFAULT_TILE"
done

echo ""
echo "RQ2 results saved to $OUT"
