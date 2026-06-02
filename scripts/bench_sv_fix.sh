#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source environment.sh

mkdir -p results /tmp/sv_fix
OUT=results/sv_fix.csv
echo "size_N,path,tile_T,time_mean_s,time_stddev_s" > "$OUT"

SIZES=(128 256 512 1024); TILE=16; WARMUP=3; RUNS=10

_compile() {
  local in=$1 out=$2
  local ll=${in}.ll
  /Users/federicobruzzone/dev/llvm-project/build/bin/mlir-translate --mlir-to-llvmir "$in" > "$ll"
  /opt/homebrew/opt/llvm/bin/clang -O3 -march=native \
    -Wno-override-module -isysroot "$(xcrun --sdk macosx --show-sdk-path)" \
    -x ir "$ll" -o "$out" \
    -L/opt/homebrew/opt/llvm/lib -lmlir_runner_utils -lmlir_c_runner_utils \
    -Wl,-rpath,/opt/homebrew/opt/llvm/lib
}

_bench() {
  local label=$1 lowered=$2 size=$3 tile=$4 niter=$5
  local bin=/tmp/sv_fix/bin_${size}_${label}
  local hfcsv=/tmp/sv_fix/hf_${size}_${label}.csv
  _compile "$lowered" "$bin"
  "$HYPERFINE" --warmup $WARMUP --runs $RUNS --export-csv "$hfcsv" "$bin" 2>/dev/null
  MEAN=$(awk -F',' -v n=$niter 'NR==2{printf "%.10f", $2/n}' "$hfcsv")
  STDDEV=$(awk -F',' -v n=$niter 'NR==2{printf "%.10f", $3/n}' "$hfcsv")
  echo "$size,$label,$tile,$MEAN,$STDDEV" | tee -a "$OUT"
}

for N in "${SIZES[@]}"; do
  case $N in 128) NITER=100;; 256) NITER=20;; 512) NITER=10;; 1024) NITER=2;; esac
  echo "=== N=$N (NITER=$NITER) ==="
  KERNEL=/tmp/sv_fix/matmul_${N}.mlir
  sed -e "s/SIZE/$N/g" -e "s/NITER/$NITER/g" kernels/matmul/bench.mlir.tpl > "$KERNEL"

  MLIR_SOURCE_BUILD=0 bash pipelines/to_affine_tiled.sh               "$KERNEL" $TILE > /tmp/sv_fix/affine_tiled_${N}.mlir
  MLIR_SOURCE_BUILD=0 bash pipelines/to_vector.sh                      "$KERNEL" $TILE > /tmp/sv_fix/vec_gather_${N}.mlir
  MLIR_SOURCE_BUILD=0 bash pipelines/to_vector_interchange.sh          "$KERNEL" $TILE > /tmp/sv_fix/vec_interchange_${N}.mlir
  MLIR_SOURCE_BUILD=1 bash pipelines/to_vector_interchange_inbounds.sh "$KERNEL" $TILE > /tmp/sv_fix/vec_inbounds_${N}.mlir

  _bench "affine_tiled"    /tmp/sv_fix/affine_tiled_${N}.mlir    $N $TILE $NITER
  _bench "vec_gather"      /tmp/sv_fix/vec_gather_${N}.mlir      $N $TILE $NITER
  _bench "vec_interchange" /tmp/sv_fix/vec_interchange_${N}.mlir $N $TILE $NITER
  _bench "vec_inbounds"    /tmp/sv_fix/vec_inbounds_${N}.mlir    $N $TILE $NITER
done

echo "Done. Results in $OUT"