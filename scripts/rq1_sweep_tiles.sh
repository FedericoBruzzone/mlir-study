#!/usr/bin/env bash
# RQ1 — Tiling Sensitivity
# Sweeps matrix sizes N and tile sizes T, records wall-clock time.
# Output: results/rq1_tiling.csv
#
# CSV columns: size_N, tile_T, time_mean_s, time_stddev_s
# Run from project root: bash scripts/rq1_sweep_tiles.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results /tmp/mlir_rq1

OUT=results/rq1_tiling.csv
echo "size_N,tile_T,time_mean_s,time_stddev_s" > "$OUT"

SIZES=(128 256 512 1024)
TILES=(16 32 64 128 256)
WARMUP=3
RUNS=10

for N in "${SIZES[@]}"; do
  # Iterations to amortize process-startup overhead (~50 ms fixed cost).
  case "$N" in
    128)  NITER=100 ;;
    256)  NITER=20  ;;
    512)  NITER=5   ;;
    1024) NITER=2   ;;
    *)    NITER=5   ;;
  esac

  KERNEL=/tmp/mlir_rq1/matmul_${N}.mlir
  sed -e "s/SIZE/$N/g" -e "s/NITER/$NITER/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

  for T in "${TILES[@]}"; do
    [ "$T" -ge "$N" ] && continue

    LOWERED=/tmp/mlir_rq1/lowered_${N}_tile${T}.mlir
    BINARY=/tmp/mlir_rq1/bin_${N}_tile${T}
    bash pipelines/to_affine_tiled.sh "$KERNEL" "$T" > "$LOWERED"
    bash scripts/_compile_native.sh "$LOWERED" "$BINARY"

    HFCSV=/tmp/mlir_rq1/hf_${N}_tile${T}.csv
    "$HYPERFINE" \
      --warmup "$WARMUP" --runs "$RUNS" \
      --export-csv "$HFCSV" \
      "$BINARY" \
      2>/dev/null

    # Divide wall-time by NITER to get per-kernel-call latency.
    MEAN=$(awk   -F',' -v n="$NITER" 'NR==2{printf "%.10f", $2/n}' "$HFCSV")
    STDDEV=$(awk -F',' -v n="$NITER" 'NR==2{printf "%.10f", $3/n}' "$HFCSV")

    echo "$N,$T,$MEAN,$STDDEV" | tee -a "$OUT"
  done
done

echo ""
echo "RQ1 results saved to $OUT"
