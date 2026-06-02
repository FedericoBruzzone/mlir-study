#!/usr/bin/env bash
# Compile a pre-lowered MLIR (LLVM dialect) to a native binary via AoT.
# Eliminates JIT overhead from timing measurements.
#
# Usage: bash scripts/_compile_native.sh <lowered.mlir> <output_binary>
# The output binary links mlir_runner_utils for vector.print support.

set -euo pipefail
source "$(dirname "$0")/../environment.sh"

LOWERED="${1:?Usage: $0 <lowered.mlir> <output_binary>}"
OUTPUT="${2:?Usage: $0 <lowered.mlir> <output_binary>}"

SDK=$(xcrun --sdk macosx --show-sdk-path)
LLVM_BIN="$(dirname "$MLIR_OPT")"
LLVM_LIB="$(dirname "$LLVM_BIN")/lib"

# BSD mktemp does not support non-X suffixes; generate base name then append .ll
TMPBASE=$(mktemp /tmp/mlir_aot_XXXXXX)
TMPLL="${TMPBASE}.ll"
trap 'rm -f "$TMPBASE" "$TMPLL"' EXIT

"$MLIR_TRANSLATE" --mlir-to-llvmir "$LOWERED" > "$TMPLL"

# Use the same LLVM toolchain as mlir-translate to avoid IR attribute mismatches
"$LLVM_BIN/clang" \
  -O3 -march=native \
  -isysroot "$SDK" \
  -Wno-override-module \
  -x ir "$TMPLL" \
  -o "$OUTPUT" \
  -L"$LLVM_LIB" \
  -lmlir_runner_utils \
  -lmlir_c_runner_utils \
  -Wl,-rpath,"$LLVM_LIB"
