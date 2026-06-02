#!/usr/bin/env bash
# Path F: Linalg → Affine → Tile → Interchange → Vectorize (NEON) → LLVM
#
# Adds --enable-loopinterchange BEFORE --affine-super-vectorize so the
# vectorizer sees j (unit-stride for B and C) as innermost instead of
# k (stride-N for B), eliminating the gather pattern on B.
#
# Problem 1 (fixed):   k innermost → gather on B (stride N) → slow
# Problem 2 (remains): vector.transfer_read on B and C lacks in_bounds
#   → masked loads (tbz + ld1.s per element) even with contiguous access.
#   --canonicalize proves in_bounds only for A (scalar broadcast), not B/C.
#   Net result: same wall-clock as to_vector.sh despite correct loop order.
#
# The correct MLIR vectorization for matmul requires linalg.vectorize or
# vector outerproduct ops, not affine-super-vectorize.
#
# Usage: bash pipelines/to_vector_interchange.sh <input.mlir> <tile_size>
# Output: lowered MLIR (LLVM dialect) on stdout

set -euo pipefail
source "$(dirname "$0")/../environment.sh"

INPUT="${1:?Usage: $0 <input.mlir> <tile_size>}"
TILE="${2:-16}"

"$MLIR_OPT" "$INPUT" \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=$TILE,$TILE,$TILE" \
  --enable-loopinterchange \
  --affine-super-vectorize="virtual-vector-size=4" \
  --canonicalize \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-ub-to-llvm \
  --convert-vector-to-llvm \
  --convert-arith-to-llvm \
  --convert-math-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --convert-index-to-llvm \
  --reconcile-unrealized-casts
