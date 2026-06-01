#!/usr/bin/env bash
# One-shot setup: creates the Python venv, builds BLAS baselines, exports model MLIR.
# Run once from the project root before any experiment.
#
# Usage: bash setup.sh
# Re-runnable: skips steps already done.

set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
LLVM_BUILD=/Users/federicobruzzone/dev/llvm-project/build

echo "=== mlir-study setup ==="

# ── 1. Python venv ─────────────────────────────────────────────────────────
if [[ ! -x "$VENV/bin/python3" ]]; then
  echo "[1/4] Creating Python venv at $VENV ..."
  python3 -m venv "$VENV"
else
  echo "[1/4] venv already exists — skipping"
fi

echo "      Installing Python deps (torch + iree-turbine)..."
"$VENV/bin/pip" install --quiet -r venv/requirements.txt

# ── 2. Validate MLIR toolchain ─────────────────────────────────────────────
echo "[2/4] Validating MLIR toolchain..."
source environment.sh   # checks tools exist, exits on missing

# Optional: switch to source-built MLIR if available
if [[ -x "$LLVM_BUILD/bin/mlir-opt" ]]; then
  echo "      Source-built MLIR found at $LLVM_BUILD/bin — set MLIR_SOURCE_BUILD=1 to use it"
fi

# ── 3. Build BLAS baselines ────────────────────────────────────────────────
echo "[3/4] Building Apple Accelerate baselines..."
$(MAKE) -C baselines all 2>/dev/null || make -C baselines all

# ── 4. Export PyTorch models → MLIR ───────────────────────────────────────
echo "[4/4] Exporting PyTorch models to MLIR via iree-turbine..."
"$VENV/bin/python3" scripts/export_models.py

echo ""
echo "=== Setup complete. Run 'make verify' to check the toolchain. ==="
echo "    Run 'make all' to reproduce all experiments."
