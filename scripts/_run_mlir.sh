#!/usr/bin/env bash
# Internal helper invoked by hyperfine — avoids commas in the timed command.
# Usage: bash scripts/_run_mlir.sh <lowered.mlir>
source "$(dirname "$0")/../environment.sh"
exec "$MLIR_RUNNER" "$1" \
  --entry-point-result=void \
  --shared-libs="$MLIR_RUNNER_LIBS" \
  --O3 --mattr=apple-m4
