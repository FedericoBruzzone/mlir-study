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
  KERNEL=/tmp/mlir_rq1/matmul_${N}.mlir
  sed "s/SIZE/$N/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

  for T in "${TILES[@]}"; do
    [ "$T" -ge "$N" ] && continue

    LOWERED=/tmp/mlir_rq1/lowered_${N}_tile${T}.mlir
    bash pipelines/to_affine_tiled.sh "$KERNEL" "$T" > "$LOWERED"

    HFCSV=/tmp/mlir_rq1/hf_${N}_tile${T}.csv
    "$HYPERFINE" \
      --warmup "$WARMUP" --runs "$RUNS" \
      --export-csv "$HFCSV" \
      "bash scripts/_run_mlir.sh $LOWERED" \
      2>/dev/null

    # CSV row 2: command,mean,stddev,...  — extract columns 2 and 3
    MEAN=$(awk -F',' 'NR==2{print $2}' "$HFCSV")
    STDDEV=$(awk -F',' 'NR==2{print $3}' "$HFCSV")

    echo "$N,$T,$MEAN,$STDDEV" | tee -a "$OUT"
  done
done

echo ""
echo "RQ1 results saved to $OUT"
