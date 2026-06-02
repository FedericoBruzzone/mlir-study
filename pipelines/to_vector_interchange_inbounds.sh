#!/usr/bin/env bash
# Path G: Linalg → Affine → Tile → Interchange → Vectorize (patched, in_bounds) → LLVM
#
# Uses MLIR_SOURCE_BUILD=1 to pick up the patched mlir-opt from
# llvm-project/build (branch sv-fix) which sets in_bounds=true on
# vector.transfer_read/write when the memref dimension is static and
# divisible by the vector width.
#
# Without this patch, --canonicalize could only prove in_bounds for
# broadcast (A) but not for B/C. The masked loads (tbz+ld1.s per element)
# remained even with correct loop order, killing performance.
#
# With this patch, all three transfer_read ops carry in_bounds=true,
# --convert-vector-to-llvm emits plain ld1/st1 instead of llvm.masked.load,
# and the result is true NEON vector code with no masking overhead.
#
# Usage: MLIR_SOURCE_BUILD=1 bash pipelines/to_vector_interchange_inbounds.sh <input.mlir> <tile_size>
# Output: lowered MLIR (LLVM dialect) on stdout

set -euo pipefail
export MLIR_SOURCE_BUILD=1
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
