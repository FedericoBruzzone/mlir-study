#!/usr/bin/env bash
# llvm-mca static throughput analysis.
# For each key variant (affine / tiled-T16 / vectorized) on matmul 512x512:
#   1. Lower to LLVM dialect via mlir-opt
#   2. Translate to LLVM IR via mlir-translate
#   3. Compile to assembly via llc (apple-m4, no PIC)
#   4. Run llvm-mca to get IPC, throughput, pipeline pressure
#
# Output: results/llvm_mca.csv  +  results/llvm_mca_<variant>.txt (full report)
# Run from project root: bash scripts/llvm_mca_analysis.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results /tmp/mlir_mca

OUT=results/llvm_mca.csv
echo "variant,N,instructions,cycles,ipc,uops_per_cycle" > "$OUT"

MCA=/Users/federicobruzzone/dev/llvm-project/build/bin/llvm-mca
LLC=/opt/homebrew/opt/llvm/bin/llc   # LLVM 23 llc not built; Homebrew llc is fine for asm gen
MCA_FLAGS="-march=aarch64 -mcpu=apple-m4 -iterations=100"

_mca_variant() {
  local label="$1" N="$2" passes="$3"

  KERNEL=/tmp/mlir_mca/matmul_${N}.mlir
  sed "s/SIZE/$N/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

  # Lower → LLVM IR → assembly
  ASM=/tmp/mlir_mca/${label}_${N}.s
  # shellcheck disable=SC2086
  "$MLIR_OPT" "$KERNEL" $passes \
    | "$MLIR_TRANSLATE" --mlir-to-llvmir \
    | "$LLC" -O3 -march=aarch64 -mcpu=apple-m4 \
        -filetype=asm -o "$ASM" 2>/dev/null

  # Isolate the hot loop body (heuristic: find the first .loop label region)
  REPORT_TXT=results/llvm_mca_${label}_${N}.txt
  "$MCA" $MCA_FLAGS "$ASM" > "$REPORT_TXT" 2>&1 || true

  # Extract summary line: "Iterations: ... Instructions: ... Total Cycles: ..."
  INSTRS=$(grep "^Instructions:" "$REPORT_TXT" | awk '{print $2}')
  CYCLES=$(grep "^Total Cycles:" "$REPORT_TXT" | awk '{print $3}')
  IPC=$(grep "^IPC:"           "$REPORT_TXT" | awk '{print $2}')
  UOPS=$(grep "^uOps Per Cycle:" "$REPORT_TXT" 2>/dev/null | awk '{print $4}' || echo "N/A")

  echo "$label,$N,${INSTRS:-N/A},${CYCLES:-N/A},${IPC:-N/A},${UOPS:-N/A}" | tee -a "$OUT"
  echo "  Full report: $REPORT_TXT"
}

echo "=== llvm-mca static analysis (apple-m4, 100 iterations) ==="

# Affine passes shared suffix
SUFFIX="--lower-affine --convert-scf-to-cf --convert-cf-to-llvm \
  --convert-vector-to-llvm --convert-arith-to-llvm --convert-math-to-llvm \
  --convert-func-to-llvm --finalize-memref-to-llvm \
  --convert-index-to-llvm --reconcile-unrealized-casts"

for N in 256 512 1024; do
  echo "--- N=$N ---"

  # Path A: untiled affine
  _mca_variant "affine_notile" "$N" \
    "--convert-linalg-to-affine-loops $SUFFIX"

  # Path C: tiled T=16 (best from RQ1)
  _mca_variant "affine_tile16" "$N" \
    "--convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=16,16,16 $SUFFIX"

  # Path D: tiled T=64
  _mca_variant "affine_tile64" "$N" \
    "--convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=64,64,64 $SUFFIX"

  # Path E: tiled T=16 + vectorized
  _mca_variant "vector_tile16" "$N" \
    "--convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=16,16,16 \
     --affine-super-vectorize=virtual-vector-size=8 $SUFFIX \
     --convert-ub-to-llvm"
done

echo ""
echo "llvm-mca results saved to $OUT"
