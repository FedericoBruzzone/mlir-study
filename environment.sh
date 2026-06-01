#!/usr/bin/env bash
# Reproducibility: pin all tool paths and document versions.
# Source this file at the top of every script: source "$(dirname "$0")/../environment.sh"
#
# MLIR_SOURCE_BUILD: set to 1 to use the local llvm-project build instead of Homebrew.
MLIR_SOURCE_BUILD="${MLIR_SOURCE_BUILD:-0}"

LLVM_HOMEBREW=/opt/homebrew/opt/llvm
LLVM_SOURCE_BIN=/Users/federicobruzzone/dev/llvm-project/build/bin

if [[ "$MLIR_SOURCE_BUILD" == "1" && -x "$LLVM_SOURCE_BIN/mlir-opt" ]]; then
  LLVM_PREFIX="$LLVM_SOURCE_BIN"
  MLIR_TOOLS="$LLVM_SOURCE_BIN"
  export MLIR_RUNNER_LIBS="$LLVM_SOURCE_BIN/../lib/libmlir_runner_utils.dylib,$LLVM_SOURCE_BIN/../lib/libmlir_c_runner_utils.dylib"
else
  LLVM_PREFIX="$LLVM_HOMEBREW"
  MLIR_TOOLS="$LLVM_HOMEBREW/bin"
  export MLIR_RUNNER_LIBS="$LLVM_HOMEBREW/lib/libmlir_runner_utils.dylib,$LLVM_HOMEBREW/lib/libmlir_c_runner_utils.dylib"
fi

export MLIR_OPT="$MLIR_TOOLS/mlir-opt"
export MLIR_RUNNER="$MLIR_TOOLS/mlir-runner"
export MLIR_TRANSLATE="$MLIR_TOOLS/mlir-translate"
export LLVM_MCA="$MLIR_TOOLS/llvm-mca"
export HYPERFINE="$(command -v hyperfine)"

# Validate tools exist
_check() { [[ -x "$1" ]] || { echo "[ERROR] missing: $1"; exit 1; }; }
_check "$MLIR_OPT"
_check "$MLIR_RUNNER"
_check "$MLIR_TRANSLATE"
_check "$HYPERFINE"
