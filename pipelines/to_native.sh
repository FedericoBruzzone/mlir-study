#!/usr/bin/env bash
# AoT compilation: Linalg MLIR → native binary via mlir-translate + clang.
# Avoids JIT overhead in timing — produces a standalone executable.
# Usage: bash pipelines/to_native.sh <input.mlir> <output_binary> [passes...]
#
# Example:
#   bash pipelines/to_native.sh kernels/matmul/bench_512.mlir /tmp/matmul_512 \
#     "--convert-linalg-to-affine-loops --affine-loop-tile=tile-sizes=64,64,64 ..."

set -euo pipefail
source "$(dirname "$0")/../environment.sh"

INPUT="${1:?Usage: $0 <input.mlir> <output_binary> [mlir-opt passes]}"
OUTPUT="${2:?Usage: $0 <input.mlir> <output_binary> [mlir-opt passes]}"
shift 2
PASSES="$*"

TMPMLIR=$(mktemp /tmp/mlir_aot_lowered_XXXXX.mlir)
TMPLL=$(mktemp /tmp/mlir_aot_XXXXX.ll)
trap 'rm -f "$TMPMLIR" "$TMPLL"' EXIT

# Step 1: lower to LLVM dialect
# shellcheck disable=SC2086
"$MLIR_OPT" "$INPUT" $PASSES > "$TMPMLIR"

# Step 2: translate to LLVM IR
"$MLIR_TRANSLATE" --mlir-to-llvmir "$TMPMLIR" > "$TMPLL"

# Step 3: compile to native binary
# -O0: do not let clang re-optimise; the MLIR passes are our optimisation.
# Link runner utils for vector.print support.
SDK=$(xcrun --sdk macosx --show-sdk-path)
LLVM_LIB_DIR="$(dirname "$(dirname "$MLIR_OPT")")/lib"
$(xcrun --find clang) -O0 -march=native \
  -isysroot "$SDK" \
  -Wno-override-module \
  -x ir "$TMPLL" \
  -o "$OUTPUT" \
  -L"$LLVM_LIB_DIR" \
  -lmlir_runner_utils \
  -lmlir_c_runner_utils \
  -Wl,-rpath,"$LLVM_LIB_DIR"
