#!/usr/bin/env bash
# Roofline analysis: compute hardware bounds + kernel positions.
# Outputs results/roofline.csv and prints a text summary.
#
# Hardware model (Apple M4 Pro, single thread):
#   Peak FP32 scalar:  ~9 GFLOP/s   (2 FLOPs/cycle × 4.5 GHz, no SIMD)
#   Peak FP32 NEON:    ~36 GFLOP/s  (4 f32 lanes × 2 FLOPs/lane (FMA) × 4.5 GHz)
#   Memory bandwidth:  ~68 GB/s     (sysctl hw.memsize / empirical)
#   Ridge point (NEON):  36e9/68e9 ≈ 0.53 FLOPs/byte
#
# Arithmetic intensity (FLOPs / bytes moved) for each kernel:
#   matmul NxN:  2N³ / (3N²×4)  = N/6  FLOPs/byte
#   conv2d 56×56×64, 3×3×64×64:  ≈ 16 FLOPs/byte
#   softmax 512×512:  ~5 / (2×4) ≈ 0.63 FLOPs/byte (near ridge)
#
# Run from project root: bash scripts/roofline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/roofline.csv
mkdir -p results

# Hardware constants (M4 Pro, single thread)
PEAK_SCALAR_GFLOPS=9.0
PEAK_NEON_GFLOPS=36.0
MEM_BW_GBS=68.0
RIDGE_SCALAR=$(awk "BEGIN{printf \"%.3f\", $PEAK_SCALAR_GFLOPS / $MEM_BW_GBS}")
RIDGE_NEON=$(awk  "BEGIN{printf \"%.3f\", $PEAK_NEON_GFLOPS  / $MEM_BW_GBS}")

echo "kernel,flops,bytes,arith_intensity,measured_gflops,roofline_bound,efficiency_pct" > "$OUT"

# Helper: compute roofline limit given arithmetic intensity (FLOPs/byte)
_roofline() {
  local ai="$1"
  awk "BEGIN{
    ai = $ai
    bw = $MEM_BW_GBS
    peak = $PEAK_NEON_GFLOPS
    roof = (ai * bw < peak) ? ai * bw : peak
    printf \"%.2f\", roof
  }"
}

# Helper: emit a row given kernel name, FLOPs, bytes moved, measured time (s)
_row() {
  local name="$1" flops="$2" bytes="$3" time_s="$4"
  awk "BEGIN{
    flops = $flops
    bytes = $bytes
    ai    = flops / bytes
    gf    = flops / ($time_s * 1e9)
    roof  = (ai * $MEM_BW_GBS < $PEAK_NEON_GFLOPS) ? ai * $MEM_BW_GBS : $PEAK_NEON_GFLOPS
    eff   = gf / roof * 100
    printf \"$name,%.0f,%.0f,%.3f,%.2f,%.2f,%.1f\n\", flops, bytes, ai, gf, roof, eff
  }" | tee -a "$OUT"
}

echo "=== Roofline Analysis — Apple M4 Pro (single thread) ==="
echo "  Peak scalar FP32 : $PEAK_SCALAR_GFLOPS GFLOP/s"
echo "  Peak NEON   FP32 : $PEAK_NEON_GFLOPS GFLOP/s"
echo "  Memory bandwidth : $MEM_BW_GBS GB/s"
echo "  Ridge point      : $RIDGE_NEON FLOPs/byte (NEON)"
echo ""
echo "kernel | AI (FLOPs/B) | measured (GFLOP/s) | roofline | efficiency"

# Matmul — use RQ2 best path (affine_tiled T=16) timings from rq2_paths.csv
if [[ -f results/rq2_paths.csv ]]; then
  for N in 128 256 512 1024; do
    T=$(awk -F',' -v n="$N" '$1==n && $2=="affine_tiled" {print $4}' results/rq2_paths.csv)
    [[ -z "$T" ]] && continue
    FLOPS=$(awk "BEGIN{printf \"%.0f\", 2.0 * $N * $N * $N}")
    BYTES=$(awk "BEGIN{printf \"%.0f\", 3.0 * $N * $N * 4}")
    _row "matmul_${N}x${N}" "$FLOPS" "$BYTES" "$T"
  done
fi

# Softmax — use RQ4 results if available
if [[ -f results/rq4_workloads.csv ]]; then
  T=$(awk -F',' '$1=="softmax" && $2=="scf" {print $3}' results/rq4_workloads.csv)
  if [[ -n "$T" ]]; then
    N=512
    # FLOPs: ~5 ops per element (max, sub, exp, sum, div), bytes: read+write=2 passes
    FLOPS=$(awk "BEGIN{printf \"%.0f\", 5.0 * $N * $N}")
    BYTES=$(awk "BEGIN{printf \"%.0f\", 2.0 * $N * $N * 4}")
    _row "softmax_${N}x${N}" "$FLOPS" "$BYTES" "$T"
  fi
fi

echo ""
echo "Roofline data saved to $OUT"
