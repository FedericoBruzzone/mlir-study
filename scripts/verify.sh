#!/usr/bin/env bash
# Smoke test: verifies the full toolchain end-to-end with a 4x4 matmul.
# Expected output: 4.000000e+00
# Run from project root: bash scripts/verify.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

TMPFILE=$(mktemp /tmp/mlir_smoke_XXXXX.mlir)
trap 'rm -f "$TMPFILE"' EXIT

echo "=== Toolchain versions ==="
"$MLIR_OPT" --version
"$HYPERFINE" --version

echo ""
echo "=== Smoke test: 4x4 matmul (Path A — affine) ==="

bash pipelines/to_affine.sh kernels/matmul/smoke_test.mlir > "$TMPFILE"

RESULT=$("$MLIR_RUNNER" "$TMPFILE" \
  --entry-point-result=void \
  --shared-libs="$MLIR_RUNNER_LIBS" 2>&1)

# vector.print format varies by LLVM version (e.g. "4" or "4.000000e+00")
# We check that the numeric value equals 4 via awk.
_check_float() {
  local result="$1" expected="$2" label="$3"
  local val
  val=$(echo "$result" | awk '{print ($1+0 == '"$expected"') ? "ok" : "fail"}')
  if [[ "$val" == "ok" ]]; then
    echo "[PASS] $label — output = $result"
  else
    echo "[FAIL] $label — expected $expected, got: $result"
    exit 1
  fi
}

_check_float "$RESULT" 4 "Path A (affine)"

echo ""
echo "=== Smoke test: 4x4 matmul (Path B — scf) ==="

bash pipelines/to_scf.sh kernels/matmul/smoke_test.mlir > "$TMPFILE"

RESULT=$("$MLIR_RUNNER" "$TMPFILE" \
  --entry-point-result=void \
  --shared-libs="$MLIR_RUNNER_LIBS" 2>&1)

_check_float "$RESULT" 4 "Path B (scf)"

echo ""
echo "All smoke tests passed."
