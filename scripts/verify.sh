#!/usr/bin/env bash
# Smoke test: verifies the full toolchain end-to-end with a 4x4 matmul.
# Uses AoT compilation (mlir-translate + clang) — consistent with benchmarks.
# Expected output: 4
# Run from project root: bash scripts/verify.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

TMPMLIR=$(mktemp /tmp/mlir_smoke_XXXXX.mlir)
TMPBIN=$(mktemp /tmp/mlir_smoke_bin_XXXXX)
trap 'rm -f "$TMPMLIR" "$TMPBIN"' EXIT

echo "=== Toolchain versions ==="
"$MLIR_OPT" --version
"$HYPERFINE" --version

# vector.print format varies by LLVM version (e.g. "4" or "4.000000e+00")
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

echo ""
echo "=== Smoke test: 4x4 matmul (Path A — affine, AoT) ==="
bash pipelines/to_affine.sh kernels/matmul/smoke_test.mlir > "$TMPMLIR"
bash scripts/_compile_native.sh "$TMPMLIR" "$TMPBIN"
RESULT=$("$TMPBIN" 2>&1)
_check_float "$RESULT" 4 "Path A (affine)"

echo ""
echo "=== Smoke test: 4x4 matmul (Path B — scf, AoT) ==="
bash pipelines/to_scf.sh kernels/matmul/smoke_test.mlir > "$TMPMLIR"
bash scripts/_compile_native.sh "$TMPMLIR" "$TMPBIN"
RESULT=$("$TMPBIN" 2>&1)
_check_float "$RESULT" 4 "Path B (scf)"

echo ""
echo "All smoke tests passed."
