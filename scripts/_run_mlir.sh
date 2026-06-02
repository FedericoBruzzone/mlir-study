#!/usr/bin/env bash
# NOT USED by any benchmark pipeline.
# All timing uses AoT-compiled binaries via scripts/_compile_native.sh.
# Kept as reference: invoking mlir-runner via hyperfine would fold
# JIT compilation and dynamic-library startup into every measured sample.
#
# Original intent: internal helper invoked by hyperfine — avoids commas in the timed command.
# Usage: bash scripts/_run_mlir.sh <lowered.mlir>
source "$(dirname "$0")/../environment.sh"
exec "$MLIR_RUNNER" "$1" \
  --entry-point-result=i32 \
  --shared-libs="$MLIR_RUNNER_LIBS" \
  --O3 --mattr=apple-m4
