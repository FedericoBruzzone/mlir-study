// Batched matmul benchmark template — (BATCH x SIZE x SIZE).
// Represents multi-head attention projections or batch inference.
// Replace BATCH and SIZE before use:
//   sed -e 's/BATCH/16/g' -e 's/SIZE/64/g' bench.mlir.tpl > /tmp/bgemm_16x64.mlir

func.func @batch_matmul(
    %A: memref<BATCHxSIZExSIZExf32>,
    %B: memref<BATCHxSIZExSIZExf32>,
    %C: memref<BATCHxSIZExSIZExf32>) {
  linalg.batch_matmul
    ins(%A, %B : memref<BATCHxSIZExSIZExf32>, memref<BATCHxSIZExSIZExf32>)
    outs(%C    : memref<BATCHxSIZExSIZExf32>)
  return
}

func.func @main() -> i32 {
  %f1     = arith.constant 1.0  : f32
  %f0     = arith.constant 0.0  : f32
  %c0     = arith.constant 0    : index
  %c1     = arith.constant 1    : index
  %niters = arith.constant NITER : index

  %A = memref.alloc() : memref<BATCHxSIZExSIZExf32>
  %B = memref.alloc() : memref<BATCHxSIZExSIZExf32>
  %C = memref.alloc() : memref<BATCHxSIZExSIZExf32>

  linalg.fill ins(%f1 : f32) outs(%A : memref<BATCHxSIZExSIZExf32>)
  linalg.fill ins(%f1 : f32) outs(%B : memref<BATCHxSIZExSIZExf32>)

  scf.for %ii = %c0 to %niters step %c1 {
    linalg.fill ins(%f0 : f32) outs(%C : memref<BATCHxSIZExSIZExf32>)
    func.call @batch_matmul(%A, %B, %C)
      : (memref<BATCHxSIZExSIZExf32>, memref<BATCHxSIZExSIZExf32>, memref<BATCHxSIZExSIZExf32>) -> ()
  }

  // C[0][0][0] = SIZE (inner dim)
  %val = memref.load %C[%c0, %c0, %c0] : memref<BATCHxSIZExSIZExf32>
  vector.print %val : f32

  memref.dealloc %A : memref<BATCHxSIZExSIZExf32>
  memref.dealloc %B : memref<BATCHxSIZExSIZExf32>
  memref.dealloc %C : memref<BATCHxSIZExSIZExf32>
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
