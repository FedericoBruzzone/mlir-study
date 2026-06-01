#!/usr/bin/env bash
# RQ-IREE: Benchmark real PyTorch models via IREE (production-quality MLIR compiler).
# This is the 4th lowering path in the study — represents state-of-the-art MLIR.
#
# Requires: venv with iree-turbine and torch (see venv/requirements.txt)
# Output: results/rq_iree.csv
#
# CSV columns: model, time_mean_ms, time_stddev_ms, gflops_est
# Run from project root: bash scripts/rq_iree.sh

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="$(dirname "$0")/../.venv"
IREE_COMPILE="$VENV/bin/iree-compile"
IREE_BENCH="$VENV/bin/iree-benchmark-module"

if [[ ! -x "$IREE_COMPILE" ]]; then
  echo "[ERROR] iree-compile not found. Run: bash setup.sh"
  exit 1
fi

mkdir -p results /tmp/mlir_iree

OUT=results/rq_iree.csv
echo "model,time_mean_ms,time_stddev_ms,gflops_est" > "$OUT"

# IREE target for Apple Silicon (AArch64, NEON-enabled)
IREE_FLAGS="--iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu-features=+neon"

IREE_RUN="$VENV/bin/iree-run-module"
HYPERFINE="$(command -v hyperfine)"

_bench_iree() {
  local name="$1" mlir="$2" flops="$3" input="$4"

  echo "  Compiling $name..."
  VMF=/tmp/mlir_iree/${name}.vmfb
  # shellcheck disable=SC2086
  if ! "$IREE_COMPILE" $IREE_FLAGS "$mlir" -o "$VMF" 2>/dev/null; then
    echo "$name,COMPILE_FAILED,N/A,N/A" | tee -a "$OUT"; return 0
  fi

  echo "  Benchmarking $name..."
  HFCSV=/tmp/mlir_iree/hf_${name}.csv
  "$HYPERFINE" \
    --warmup 5 --runs 20 \
    --export-csv "$HFCSV" \
    "$IREE_RUN --device=local-task --module=$VMF --function=main --input=$input" \
    2>/dev/null

  MEAN=$(awk   -F',' 'NR==2{print $2}' "$HFCSV")
  STDDEV=$(awk -F',' 'NR==2{print $3}' "$HFCSV")
  GF=$(awk "BEGIN{printf \"%.2f\", $flops / ($MEAN * 1e9)}")

  echo "$name,$MEAN,$STDDEV,$GF" | tee -a "$OUT"
}

echo "=== IREE benchmarks (production-quality MLIR) ==="

# Real PyTorch model layers (exported by scripts/export_models.py)
# FLOPs estimated: linear 512→512: 2*512*512 = 524k; BERT MHA: ~2*128*768*768*3 ≈ 339M
_bench_iree "linear_relu_512"    kernels/models/linear_relu_512.mlir    524288    "1x512xf32"
_bench_iree "linear_relu_1024"   kernels/models/linear_relu_1024.mlir   2097152   "1x1024xf32"
_bench_iree "conv_bn_relu_56x56" kernels/models/conv_bn_relu_56x56.mlir 231211008 "1x64x56x56xf32"
_bench_iree "mha_bert_base"      kernels/models/mha_bert_base.mlir      339738624 "1x128x768xf32"
_bench_iree "mobile_block_56x56" kernels/models/mobile_block_56x56.mlir 28901376  "1x32x56x56xf32"

echo ""
echo "IREE results saved to $OUT"
