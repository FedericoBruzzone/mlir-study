// 4x4 all-ones matmul — correctness check.
// A[i,k]=1, B[k,j]=1 → C[i,j]=4 (inner dim=4).
// Expected output: 4.000000e+00

func.func @main() {
  %f1 = arith.constant 1.0 : f32
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  %A = memref.alloc() : memref<4x4xf32>
  %B = memref.alloc() : memref<4x4xf32>
  %C = memref.alloc() : memref<4x4xf32>

  linalg.fill ins(%f1 : f32) outs(%A : memref<4x4xf32>)
  linalg.fill ins(%f1 : f32) outs(%B : memref<4x4xf32>)
  linalg.fill ins(%f0 : f32) outs(%C : memref<4x4xf32>)

  linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
               outs(%C : memref<4x4xf32>)

  %val = memref.load %C[%c0, %c0] : memref<4x4xf32>
  vector.print %val : f32

  memref.dealloc %A : memref<4x4xf32>
  memref.dealloc %B : memref<4x4xf32>
  memref.dealloc %C : memref<4x4xf32>
  return
}
