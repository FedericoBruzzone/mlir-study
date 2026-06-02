// Row-wise softmax on a 512x512 tensor — memory-bandwidth dominated.
// Used as a contrast to the compute-bound matmul for roofline positioning.
//
// Algorithm: for each row i:
//   1. max_val  = max(x[i, :])
//   2. shifted  = x[i, :] - max_val   (numerical stability)
//   3. exp_vals = exp(shifted[i, :])
//   4. sum_exp  = sum(exp_vals[i, :])
//   5. out[i, :] = exp_vals[i, :] / sum_exp

func.func @softmax(%x: memref<512x512xf32>, %out: memref<512x512xf32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %neginf = arith.constant 0xFF800000 : i32  // -inf bit pattern

  scf.for %i = %c0 to %c512 step %c1 {
    // Step 1: row max
    %neg_inf_f = arith.bitcast %neginf : i32 to f32
    %max_val = scf.for %j = %c0 to %c512 step %c1 iter_args(%acc = %neg_inf_f) -> f32 {
      %v = memref.load %x[%i, %j] : memref<512x512xf32>
      %m = arith.maximumf %acc, %v : f32
      scf.yield %m : f32
    }
    // Step 2+3: shifted exp
    %sum = scf.for %j = %c0 to %c512 step %c1 iter_args(%s = %neg_inf_f) -> f32 {
      %v  = memref.load %x[%i, %j] : memref<512x512xf32>
      %sh = arith.subf %v, %max_val : f32
      %e  = math.exp %sh : f32
      memref.store %e, %out[%i, %j] : memref<512x512xf32>
      %ns = arith.addf %s, %e : f32
      scf.yield %ns : f32
    }
    // Step 4: normalise
    scf.for %j = %c0 to %c512 step %c1 {
      %e  = memref.load %out[%i, %j] : memref<512x512xf32>
      %r  = arith.divf %e, %sum : f32
      memref.store %r, %out[%i, %j] : memref<512x512xf32>
    }
  }
  return
}

func.func @main() -> i32 {
  %f1     = arith.constant 1.0 : f32
  %f0     = arith.constant 0.0 : f32
  %c0     = arith.constant 0   : index
  %c1     = arith.constant 1   : index
  %niters = arith.constant 20  : index

  %x   = memref.alloc() : memref<512x512xf32>
  %out = memref.alloc() : memref<512x512xf32>

  linalg.fill ins(%f1 : f32) outs(%x : memref<512x512xf32>)

  scf.for %ii = %c0 to %niters step %c1 {
    linalg.fill ins(%f0 : f32) outs(%out : memref<512x512xf32>)
    func.call @softmax(%x, %out) : (memref<512x512xf32>, memref<512x512xf32>) -> ()
  }

  // All inputs equal → each output element = 1/512 ≈ 0.001953
  %val = memref.load %out[%c0, %c0] : memref<512x512xf32>
  vector.print %val : f32

  memref.dealloc %x   : memref<512x512xf32>
  memref.dealloc %out : memref<512x512xf32>
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
