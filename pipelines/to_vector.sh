#!/usr/bin/env bash
# Path D: Linalg → Affine → Tile → Vectorize (NEON) → LLVM
# Usage: bash pipelines/to_vector.sh <input.mlir> <tile_size>
# Output: lowered MLIR (LLVM dialect) on stdout

set -euo pipefail
source "$(dirname "$0")/../environment.sh"

INPUT="${1:?Usage: $0 <input.mlir> <tile_size>}"
TILE="${2:-64}"

"$MLIR_OPT" "$INPUT" \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=$TILE,$TILE,$TILE" \
  --affine-super-vectorize="virtual-vector-size=4" \
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
