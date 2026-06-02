#!/usr/bin/env bash
# Diagnose why affine-super-vectorize underperforms LLVM auto-vec on matmul.
#
# Generates and saves four IR levels for three paths at N=512, T=16:
#   *.vec.mlir        — MLIR after affine-super-vectorize (vector dialect ops)
#   *.llvmir.mlir     — MLIR LLVM dialect (just before mlir-translate)
#   *.ll              — LLVM IR (output of mlir-translate --mlir-to-llvmir)
#   *.s               — AArch64 assembly (output of llc -O3)
#
# Three paths compared:
#   Path A — affine_tiled        : tile T=16, LLVM decides (scalar, no gather)
#   Path B — vec                 : tile + affine-super-vectorize (gather on k)
#   Path C — vec_interchange     : tile + interchange + super-vectorize + canonicalize
#
# Two distinct problems diagnosed:
#   Problem 1: affine-super-vectorize picks the k loop (stride-N gather on B)
#              → fixed by --enable-loopinterchange (j becomes innermost)
#   Problem 2: vector.transfer_read on B and C lacks in_bounds
#              → masked loads (tbz + ld1.s) even with correct loop order
#              → --canonicalize adds in_bounds only for A (broadcast), not B/C
#              → net result: vec_interchange ≈ vec in wall-clock time
#
# Run from project root: bash scripts/diagnose_vec.sh
# Output directory: /tmp/diag/

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

N=512
TILE=16
LLC=/opt/homebrew/opt/llvm/bin/llc
LLC_FLAGS="-O3 -march=aarch64 -mcpu=apple-m4 -filetype=asm"
OUT=/tmp/diag
mkdir -p "$OUT"

echo "=== Generating IR levels for N=$N, T=$TILE ==="
echo ""

KERNEL=$OUT/kernel_${N}.mlir
sed -e "s/SIZE/$N/g" -e "s/NITER/1/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

# ── Path A: affine tiled, no explicit vectorization ──────────────────────────
LLVMIR_A=$OUT/affine_tiled.llvmir.mlir ; LL_A=$OUT/affine_tiled.ll ; ASM_A=$OUT/affine_tiled.s
bash pipelines/to_affine_tiled.sh "$KERNEL" "$TILE" > "$LLVMIR_A"
"$MLIR_TRANSLATE" --mlir-to-llvmir "$LLVMIR_A" > "$LL_A"
"$LLC" $LLC_FLAGS "$LL_A" -o "$ASM_A"

# ── Path B: affine-super-vectorize (k is innermost → gather) ─────────────────
VEC_B=$OUT/vec.vec.mlir ; LLVMIR_B=$OUT/vec.llvmir.mlir ; LL_B=$OUT/vec.ll ; ASM_B=$OUT/vec.s
"$MLIR_OPT" "$KERNEL" \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=$TILE,$TILE,$TILE" \
  --affine-super-vectorize="virtual-vector-size=4" \
  > "$VEC_B"
bash pipelines/to_vector.sh "$KERNEL" "$TILE" > "$LLVMIR_B"
"$MLIR_TRANSLATE" --mlir-to-llvmir "$LLVMIR_B" > "$LL_B"
"$LLC" $LLC_FLAGS "$LL_B" -o "$ASM_B"

# ── Path C: interchange + affine-super-vectorize + canonicalize ───────────────
VEC_C=$OUT/vec_interchange.vec.mlir
VEC_C_CANON=$OUT/vec_interchange.canonicalized.mlir
LLVMIR_C=$OUT/vec_interchange.llvmir.mlir ; LL_C=$OUT/vec_interchange.ll ; ASM_C=$OUT/vec_interchange.s

# MLIR after interchange+vectorize (before canonicalize) — shows loop order
"$MLIR_OPT" "$KERNEL" \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=$TILE,$TILE,$TILE" \
  --enable-loopinterchange \
  --affine-super-vectorize="virtual-vector-size=4" \
  > "$VEC_C"
# MLIR after canonicalize — shows which transfer_reads get in_bounds
"$MLIR_OPT" "$KERNEL" \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=$TILE,$TILE,$TILE" \
  --enable-loopinterchange \
  --affine-super-vectorize="virtual-vector-size=4" \
  --canonicalize \
  > "$VEC_C_CANON"
bash pipelines/to_vector_interchange.sh "$KERNEL" "$TILE" > "$LLVMIR_C"
"$MLIR_TRANSLATE" --mlir-to-llvmir "$LLVMIR_C" > "$LL_C"
"$LLC" $LLC_FLAGS "$LL_C" -o "$ASM_C"

printf "Generated in %s/:\n" "$OUT"
for f in affine_tiled.llvmir.mlir affine_tiled.ll affine_tiled.s \
          vec.vec.mlir vec.llvmir.mlir vec.ll vec.s \
          vec_interchange.vec.mlir vec_interchange.canonicalized.mlir \
          vec_interchange.llvmir.mlir vec_interchange.ll vec_interchange.s; do
  printf "  %-42s %5d lines\n" "$f" "$(wc -l < "$OUT/$f")"
done
echo ""

section() { echo ""; printf "%-52s\n" "" | tr ' ' '─'; echo "$1"; printf "%-52s\n" "" | tr ' ' '─'; }

# ── 0. MLIR vector ops — key difference between B and C ──────────────────────
section "0. MLIR VECTOR OPS: PATH B vs PATH C"

echo "[Path B] vec.vec.mlir — k is innermost (WRONG):"
echo "  Inner loop order after tile (no interchange):"
grep "affine.for" "$VEC_B" | tail -6 | head -3 | sed 's/^/  /'
echo "  transfer_read indices: A[i,k], B[k,j], C[i,j]"
grep "transfer_read" "$VEC_B" | head -3 | sed 's/^/    /'

echo ""
echo "[Path C] vec_interchange.vec.mlir — j is innermost (CORRECT loop order):"
echo "  Inner loop order after interchange:"
grep "affine.for" "$VEC_C" | tail -6 | head -3 | sed 's/^/  /'
echo "  transfer_read indices (j vectorized, B[k,j:j+4] contiguous):"
grep "transfer_read" "$VEC_C" | head -3 | sed 's/^/    /'

echo ""
echo "[Path C] vec_interchange.canonicalized.mlir — in_bounds after canonicalize:"
echo "  Which transfer_reads get in_bounds=true?"
grep "transfer_read" "$VEC_C_CANON" | sed 's/^/    /'
echo "  NOTE: B and C still lack in_bounds → masked loads remain"

# ── 1. FP instruction mix ────────────────────────────────────────────────────
section "1. FP INSTRUCTION MIX"

count_fp() {
  local label="$1" f="$2"
  local s_mul s_add v_mul v_add v_ins
  s_mul=$(grep -cE "fmul\s+s|fmadd\s+s" "$f" || true)
  s_add=$(grep -cE "fadd\s+s"            "$f" || true)
  v_mul=$(grep -cE "fmul\.[248]s"        "$f" || true)
  v_add=$(grep -cE "fadd\.[248]s"        "$f" || true)
  v_ins=$(grep -cE "ld1\.s"              "$f" || true)
  tbz=$(grep -cE "tbz|tbnz"              "$f" || true)
  printf "  %-28s scalar(mul+add): %d+%d   vector(mul+add): %d+%d   scatter/gather inserts: %d   mask branches: %d\n" \
    "$label" "$s_mul" "$s_add" "$v_mul" "$v_add" "$v_ins" "$tbz"
}
count_fp "Path A  affine_tiled.s"    "$ASM_A"
count_fp "Path B  vec.s"              "$ASM_B"
count_fp "Path C  vec_interchange.s" "$ASM_C"

# ── 2. Hot inner loop body ────────────────────────────────────────────────────
section "2. HOT INNER LOOP BODY"

echo "[Path A] scalar — LLVM chose NOT to vectorize (stride-N on B detected):"
grep -B2 -A10 "fmul	s\|fadd	s\|fmadd	s" "$ASM_A" 2>/dev/null | head -20 || echo "  (not found)"

echo ""
echo "[Path B] gather — affine-super-vectorize forced k-loop SIMD:"
grep -B1 -A14 "ld1\.s" "$ASM_B" 2>/dev/null | head -25 || echo "  (not found)"

echo ""
echo "[Path C] masked gather — loop order fixed but in_bounds still missing:"
grep -B1 -A14 "ld1\.s\|tbz" "$ASM_C" 2>/dev/null | head -25 || echo "  (not found)"

# ── 3. Benchmark ─────────────────────────────────────────────────────────────
section "3. BENCHMARK N=$N, T=$TILE (NITER=5)"

NITER=5
sed -e "s/SIZE/$N/g" -e "s/NITER/$NITER/g" kernels/matmul/bench.mlir.tpl > /tmp/bench_diag.mlir

bash pipelines/to_affine_tiled.sh   /tmp/bench_diag.mlir "$TILE" > /tmp/b_a.mlir
bash pipelines/to_vector.sh          /tmp/bench_diag.mlir "$TILE" > /tmp/b_b.mlir
bash pipelines/to_vector_interchange.sh /tmp/bench_diag.mlir "$TILE" > /tmp/b_c.mlir

bash scripts/_compile_native.sh /tmp/b_a.mlir /tmp/bin_diag_a
bash scripts/_compile_native.sh /tmp/b_b.mlir /tmp/bin_diag_b
bash scripts/_compile_native.sh /tmp/b_c.mlir /tmp/bin_diag_c

HYPERFINE_BIN="$(command -v hyperfine)"
"$HYPERFINE_BIN" \
  --warmup 3 --runs 10 \
  --command-name "A: affine_tiled (scalar)"          "/tmp/bin_diag_a" \
  --command-name "B: vec (gather, k-loop)"           "/tmp/bin_diag_b" \
  --command-name "C: vec_interchange (masked loads)" "/tmp/bin_diag_c" \
  2>&1 | grep -E "Benchmark|Time|faster|slower"

# ── 4. Root cause summary ─────────────────────────────────────────────────────
section "4. ROOT CAUSE SUMMARY"
cat <<'EOF'
PATH A — affine_tiled (to_affine_tiled.sh)
  LLVM generates scalar FMA. It detects stride-N on B and refuses to
  vectorize the k loop. M4 OOO engine pipelines the scalar ops efficiently.

PATH B — vec (to_vector.sh)
  PROBLEM 1: affine-super-vectorize picks the innermost loop (k).
    B[k,j] has stride N → 4 scalar inserts (ld1.s) to build a 4-wide vector.
    Cost: 4 scalar loads + insert overhead per vector op = slower than scalar.

PATH C — vec_interchange (to_vector_interchange.sh)
  PROBLEM 1 fixed: --enable-loopinterchange makes j innermost.
    B[k,j:j+4] is now contiguous → correct for SIMD.
  PROBLEM 2 remains: vector.transfer_read on B and C lacks {in_bounds=[true]}.
    --canonicalize adds in_bounds only for A (scalar broadcast, trivially safe).
    B and C bounds depend on outer tile variables → canonicalize cannot prove them.
    Result: masked loads (tbz/tbnz + ld1.s per element) survive → same cost as Path B.
    Upstream issue: https://discourse.llvm.org/t/mlir-affine-affine-super-vectorize-does-not-set-in-bounds-on-transfer-ops-for-statically-divisible-shapes/90785

WHY LLVM IS BETTER:
  LLVM's TargetTransformInfo cost model knows that a gather (stride-N load)
  on AArch64 costs more than equivalent scalar ops. It chooses NOT to vectorize
  the k loop, avoiding both the gather and the masking overhead.

CORRECT VECTORIZATION FOR MATMUL IN MLIR:
  Use linalg.vectorize (operates before tiling, in-bounds guaranteed by
  construction) or vector.outerproduct ops.
  affine-super-vectorize is not designed for non-trivial stride patterns.

OUTPUT FILES:
  vec.vec.mlir                ← Path B: transfer_read with gather map
  vec_interchange.vec.mlir    ← Path C: correct loop order, no in_bounds on B/C
  vec_interchange.canonicalized.mlir  ← Path C: in_bounds=true only on A
  *.ll                        ← LLVM IR: Path B/C have insertelement chains
  *.s                         ← Assembly: Path A scalar, B/C ld1.s scatter
EOF

echo ""
echo "All files in $OUT/"
