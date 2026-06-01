#!/usr/bin/env bash
# Dumps hardware and tool version info to results/environment.txt.
# Run once before experiments: bash scripts/collect_env.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results
OUT=results/environment.txt

{
  echo "=== Date ==="
  date -u

  echo ""
  echo "=== macOS ==="
  sw_vers

  echo ""
  echo "=== CPU ==="
  sysctl -n machdep.cpu.brand_string  2>/dev/null || sysctl -n hw.model
  echo "Physical cores : $(sysctl -n hw.physicalcpu)"
  echo "Logical  cores : $(sysctl -n hw.logicalcpu)"
  echo "L1 D-cache (B) : $(sysctl -n hw.l1dcachesize)"
  echo "L2   cache (B) : $(sysctl -n hw.l2cachesize)"
  echo "RAM        (B) : $(sysctl -n hw.memsize)"

  echo ""
  echo "=== MLIR / LLVM ==="
  "$MLIR_OPT" --version

  echo ""
  echo "=== hyperfine ==="
  "$HYPERFINE" --version
} | tee "$OUT"

echo ""
echo "Saved to $OUT"
