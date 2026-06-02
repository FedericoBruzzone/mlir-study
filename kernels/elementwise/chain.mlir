// Elementwise chain: matmul → relu → bias_add (512x512, FP32).
// Used for RQ3 (fusion impact).
// Compile with --affine-loop-fusion to get the fused variant.
//
// Input layout:
//   A, B  : 512x512 (matmul operands)
//   bias  : 512     (column bias, broadcast over rows)
//   C     : 512x512 (matmul output / relu input buffer)
//   out   : 512x512 (final output)

func.func @chain(%A:    memref<512x512xf32>,
                 %B:    memref<512x512xf32>,
                 %bias: memref<512xf32>,
                 %C:    memref<512x512xf32>,
                 %out:  memref<512x512xf32>) {
  // Step 1 — matmul: C = A × B
  linalg.matmul ins(%A, %B : memref<512x512xf32>, memref<512x512xf32>)
               outs(%C : memref<512x512xf32>)

  // Step 2 — relu: out = max(C, 0)
  %zero = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%C : memref<512x512xf32>) outs(%out : memref<512x512xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.maximumf %in, %zero : f32
      linalg.yield %r : f32
  }

  // Step 3 — bias add: out += bias[j]
  linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias : memref<512xf32>) outs(%out : memref<512x512xf32>) {
    ^bb0(%b: f32, %acc: f32):
      %s = arith.addf %acc, %b : f32
      linalg.yield %s : f32
  }

  return
}

func.func @main() -> i32 {
  %f1     = arith.constant 1.0 : f32
  %f0     = arith.constant 0.0 : f32
  %c0     = arith.constant 0   : index
  %c1     = arith.constant 1   : index
  %niters = arith.constant 5   : index

  %A    = memref.alloc() : memref<512x512xf32>
  %B    = memref.alloc() : memref<512x512xf32>
  %bias = memref.alloc() : memref<512xf32>
  %C    = memref.alloc() : memref<512x512xf32>
  %out  = memref.alloc() : memref<512x512xf32>

  linalg.fill ins(%f1 : f32) outs(%A    : memref<512x512xf32>)
  linalg.fill ins(%f1 : f32) outs(%B    : memref<512x512xf32>)
  linalg.fill ins(%f1 : f32) outs(%bias : memref<512xf32>)

  scf.for %ii = %c0 to %niters step %c1 {
    linalg.fill ins(%f0 : f32) outs(%C   : memref<512x512xf32>)
    linalg.fill ins(%f0 : f32) outs(%out : memref<512x512xf32>)
    func.call @chain(%A, %B, %bias, %C, %out)
      : (memref<512x512xf32>, memref<512x512xf32>, memref<512xf32>,
         memref<512x512xf32>, memref<512x512xf32>) -> ()
  }

  // out[0][0] = max(512.0, 0) + 1.0 = 513.0
  %val = memref.load %out[%c0, %c0] : memref<512x512xf32>
  vector.print %val : f32

  memref.dealloc %A    : memref<512x512xf32>
  memref.dealloc %B    : memref<512x512xf32>
  memref.dealloc %bias : memref<512xf32>
  memref.dealloc %C    : memref<512x512xf32>
  memref.dealloc %out  : memref<512x512xf32>
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
