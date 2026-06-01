// NxN FP32 matmul benchmark template.
// Replace SIZE with the matrix dimension before use:
//   sed 's/SIZE/512/g' bench.mlir.tpl > /tmp/matmul_512.mlir
//
// A[i,k]=1, B[k,j]=1 → C[i,j]=SIZE (used as correctness guard).

func.func @matmul(%A: memref<SIZExSIZExf32>,
                  %B: memref<SIZExSIZExf32>,
                  %C: memref<SIZExSIZExf32>) {
  linalg.matmul ins(%A, %B : memref<SIZExSIZExf32>, memref<SIZExSIZExf32>)
               outs(%C : memref<SIZExSIZExf32>)
  return
}

func.func @main() {
  %f1  = arith.constant 1.0 : f32
  %f0  = arith.constant 0.0 : f32
  %c0  = arith.constant 0 : index

  %A = memref.alloc() : memref<SIZExSIZExf32>
  %B = memref.alloc() : memref<SIZExSIZExf32>
  %C = memref.alloc() : memref<SIZExSIZExf32>

  linalg.fill ins(%f1 : f32) outs(%A : memref<SIZExSIZExf32>)
  linalg.fill ins(%f1 : f32) outs(%B : memref<SIZExSIZExf32>)
  linalg.fill ins(%f0 : f32) outs(%C : memref<SIZExSIZExf32>)

  call @matmul(%A, %B, %C)
    : (memref<SIZExSIZExf32>, memref<SIZExSIZExf32>, memref<SIZExSIZExf32>) -> ()

  %val = memref.load %C[%c0, %c0] : memref<SIZExSIZExf32>
  vector.print %val : f32

  memref.dealloc %A : memref<SIZExSIZExf32>
  memref.dealloc %B : memref<SIZExSIZExf32>
  memref.dealloc %C : memref<SIZExSIZExf32>
  return
}
