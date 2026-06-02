#!/usr/bin/env bash
# Diagnose: affine-super-vectorize in_bounds fix (sv-fix branch)
set -euo pipefail
cd "$(dirname "$0")/.."

N=512; TILE=16
LLC_FLAGS="-O3 -march=aarch64 -mcpu=apple-m4 -filetype=asm"
OUT=/tmp/sv_fix_diag
mkdir -p "$OUT"

MLIR_OPT_UNPAT="/opt/homebrew/opt/llvm/bin/mlir-opt"
MLIR_OPT_PAT="/Users/federicobruzzone/dev/llvm-project/build/bin/mlir-opt"
MLIR_TRANSLATE="/Users/federicobruzzone/dev/llvm-project/build/bin/mlir-translate"
LLC="/opt/homebrew/opt/llvm/bin/llc"
SV_FLAGS="--convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=$TILE,$TILE,$TILE --enable-loopinterchange --affine-super-vectorize=virtual-vector-size=4"

KERNEL=$OUT/kernel_${N}.mlir
sed -e "s/SIZE/$N/g" -e "s/NITER/1/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

section() { echo ""; printf "%-40s\n" "" | tr ' ' '─'; echo ">>> $1"; printf "%-40s\n" "" | tr ' ' '─'; }

# ── Path C: vec_interchange (unpatched) ────────────────────────────────────
VEC_C=$OUT/c_vec_interchange.vec.mlir
VEC_C_CANON=$OUT/c_vec_interchange.canonicalized.mlir
LLVMIR_C=$OUT/c_vec_interchange.llvmir.mlir; ASM_C=$OUT/c_vec_interchange.s
"$MLIR_OPT_UNPAT" "$KERNEL" $SV_FLAGS > "$VEC_C"
"$MLIR_OPT_UNPAT" "$KERNEL" $SV_FLAGS --canonicalize > "$VEC_C_CANON"
source environment.sh && MLIR_SOURCE_BUILD=0 bash pipelines/to_vector_interchange.sh "$KERNEL" "$TILE" > "$LLVMIR_C"
"$MLIR_TRANSLATE" --mlir-to-llvmir "$LLVMIR_C" | "$LLC" $LLC_FLAGS -o "$ASM_C" -

# ── Path D: vec_inbounds (patched) ────────────────────────────────────────
VEC_D=$OUT/d_vec_inbounds.vec.mlir
VEC_D_CANON=$OUT/d_vec_inbounds.canonicalized.mlir
LLVMIR_D=$OUT/d_vec_inbounds.llvmir.mlir; ASM_D=$OUT/d_vec_inbounds.s
"$MLIR_OPT_PAT" "$KERNEL" $SV_FLAGS > "$VEC_D"
"$MLIR_OPT_PAT" "$KERNEL" $SV_FLAGS --canonicalize > "$VEC_D_CANON"
MLIR_SOURCE_BUILD=1 bash pipelines/to_vector_interchange_inbounds.sh "$KERNEL" "$TILE" > "$LLVMIR_D"
"$MLIR_TRANSLATE" --mlir-to-llvmir "$LLVMIR_D" | "$LLC" $LLC_FLAGS -o "$ASM_D" -

section "0. in_bounds BEFORE canonicalize (raw affine-super-vectorize output)"
echo "[C - unpatched] B and C lack in_bounds:"; grep "transfer_read" "$VEC_C" | sed 's/^/  /'
echo "[D - patched]   all three have in_bounds:"; grep "transfer_read" "$VEC_D" | sed 's/^/  /'

section "0b. in_bounds AFTER canonicalize"
echo "[C - unpatched] only A (broadcast) gets in_bounds:"; grep "transfer_read" "$VEC_C_CANON" | sed 's/^/  /'
echo "[D - patched]   all three retain in_bounds:"; grep "transfer_read" "$VEC_D_CANON" | sed 's/^/  /'

section "1. MASKED LOADS in assembly"
echo "[C] tbz count: $(grep -c "tbz\|tbnz" "$ASM_C" 2>/dev/null || echo 0)"
grep -n "tbz\|ld1\.s\[" "$ASM_C" | head -12 | sed 's/^/  /'
echo ""
echo "[D] tbz count: $(grep -c "tbz\|tbnz" "$ASM_D" 2>/dev/null || echo 0)"
grep -n "ld1\s\+{v" "$ASM_D" | head -6 | sed 's/^/  /'

# ── Benchmark: all 4 paths ─────────────────────────────────────────────────
section "2. BENCHMARK N=$N, T=$TILE (NITER=10)"

sed -e "s/SIZE/$N/g" -e "s/NITER/10/g" kernels/matmul/bench.mlir.tpl > /tmp/bench_diag.mlir
MLIR_SOURCE_BUILD=0 bash pipelines/to_affine_tiled.sh               /tmp/bench_diag.mlir "$TILE" > /tmp/b_a.mlir
MLIR_SOURCE_BUILD=0 bash pipelines/to_vector.sh                      /tmp/bench_diag.mlir "$TILE" > /tmp/b_b.mlir
MLIR_SOURCE_BUILD=0 bash pipelines/to_vector_interchange.sh          /tmp/bench_diag.mlir "$TILE" > /tmp/b_c.mlir
MLIR_SOURCE_BUILD=1 bash pipelines/to_vector_interchange_inbounds.sh /tmp/bench_diag.mlir "$TILE" > /tmp/b_d.mlir

MLIR_SOURCE_BUILD=1 bash scripts/_compile_native.sh /tmp/b_a.mlir /tmp/bin_a
MLIR_SOURCE_BUILD=1 bash scripts/_compile_native.sh /tmp/b_b.mlir /tmp/bin_b
MLIR_SOURCE_BUILD=1 bash scripts/_compile_native.sh /tmp/b_c.mlir /tmp/bin_c
MLIR_SOURCE_BUILD=1 bash scripts/_compile_native.sh /tmp/b_d.mlir /tmp/bin_d

"$(command -v hyperfine)" \
  --warmup 3 --runs 10 \
  --command-name "A: affine_tiled (scalar)"          "/tmp/bin_a" \
  --command-name "B: vec (gather)"                   "/tmp/bin_b" \
  --command-name "C: vec_interchange (masked loads)" "/tmp/bin_c" \
  --command-name "D: vec_inbounds (fix)"             "/tmp/bin_d"

section "3. SUMMARY"
echo "A: scalar LLVM auto-vec (baseline)"
echo "B: k-loop gather (stride-N, always slow)"
echo "C: j-loop correct but masked loads (tbz+ld1.s scatter)"
echo "D: j-loop + in_bounds[true] -> plain ld1 {v.4s}"
echo "EXPECT: D faster than C; D vs A tells if NEON beats OOO-scalar"
