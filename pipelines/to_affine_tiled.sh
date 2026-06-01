#!/usr/bin/env bash
# Path C: Linalg → Affine loops → Tiling → LLVM dialect
# Usage: bash pipelines/to_affine_tiled.sh <input.mlir> <tile_size>
# Output: lowered MLIR (LLVM dialect) on stdout

set -euo pipefail
source "$(dirname "$0")/../environment.sh"

INPUT="${1:?Usage: $0 <input.mlir> <tile_size>}"
TILE="${2:?Usage: $0 <input.mlir> <tile_size>}"

"$MLIR_OPT" "$INPUT" \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=$TILE,$TILE,$TILE" \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-vector-to-llvm \
  --convert-arith-to-llvm \
  --convert-math-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --convert-index-to-llvm \
  --reconcile-unrealized-casts