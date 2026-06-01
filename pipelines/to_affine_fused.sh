#!/usr/bin/env bash
# Path D: Linalg → Affine loops → Loop fusion → LLVM dialect  (RQ3)
# Usage: bash pipelines/to_affine_fused.sh <input.mlir>
# Output: lowered MLIR (LLVM dialect) on stdout

set -euo pipefail
source "$(dirname "$0")/../environment.sh"

INPUT="${1:?Usage: $0 <input.mlir>}"

"$MLIR_OPT" "$INPUT" \
  --convert-linalg-to-affine-loops \
  --affine-loop-fusion \
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
