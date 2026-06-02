#!/usr/bin/env bash
# RQ-IREE: Benchmark real PyTorch models via IREE (production-quality MLIR compiler).
# This is the 4th lowering path in the study — represents state-of-the-art MLIR.
#
# Requires: venv with iree-turbine and torch (see venv/requirements.txt)
# Output: results/rq_iree.csv
#
# CSV columns: model, time_mean_s, time_stddev_s, gflops_est
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
echo "model,time_mean_s,time_stddev_s,gflops_est" > "$OUT"

# IREE target for Apple Silicon (AArch64, NEON-enabled)
IREE_FLAGS="--iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu-features=+neon"

WARMUP=20
RUNS=200

_bench_iree() {
  local name="$1" mlir="$2" flops="$3" input="$4"

  echo "  Compiling $name..."
  VMF=/tmp/mlir_iree/${name}.vmfb
  # shellcheck disable=SC2086
  if ! "$IREE_COMPILE" $IREE_FLAGS "$mlir" -o "$VMF" 2>/dev/null; then
    echo "$name,COMPILE_FAILED,N/A,N/A" | tee -a "$OUT"; return 0
  fi

  echo "  Benchmarking $name..."
  BENCH_CSV=/tmp/mlir_iree/bench_${name}.csv
  # iree-benchmark-module runs in-process — no per-invocation startup overhead.
  # real_time in the CSV is per-call latency in ms.
  "$IREE_BENCH" \
    --device=local-task \
    --module="$VMF" \
    --function=main \
    --input="$input" \
    --benchmark_min_warmup_time="$WARMUP" \
    --benchmark_repetitions="$RUNS" \
    --benchmark_out_format=csv \
    --benchmark_out="$BENCH_CSV" \
    > /dev/null 2>&1

  # CSV row format: "name",iterations,real_time,cpu_time,time_unit,...
  # Aggregate rows end in _mean / _stddev; real_time is in ms.
  MEAN_MS=$(awk  -F',' '/_mean/{  gsub(/"/, "", $1); print $3; exit}' "$BENCH_CSV")
  STDDEV_MS=$(awk -F',' '/_stddev/{gsub(/"/, "", $1); print $3; exit}' "$BENCH_CSV")
  MEAN=$(awk   "BEGIN{printf \"%.10f\", $MEAN_MS   / 1000}")
  STDDEV=$(awk "BEGIN{printf \"%.10f\", $STDDEV_MS / 1000}")
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
