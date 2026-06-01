#!/usr/bin/env bash
# Path B: Linalg → SCF loops → LLVM dialect
# Usage: bash pipelines/to_scf.sh <input.mlir>
# Output: lowered MLIR (LLVM dialect) on stdout

set -euo pipefail
source "$(dirname "$0")/../environment.sh"

INPUT="${1:?Usage: $0 <input.mlir>}"

"$MLIR_OPT" "$INPUT" \
  --convert-linalg-to-loops \
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
