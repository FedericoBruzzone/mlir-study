// 2D convolution benchmark template — NHWC layout.
// Replace SIZE with the spatial dimension (SIZExSIZE feature map).
// Filter fixed: 3x3x64x64 (ResNet-like, 64 input/output channels).
// Output spatial: (SIZE-2) x (SIZE-2) — valid padding.
//
// Usage: sed -e 's/SIZE/56/g' -e 's/OSIZE/54/g' bench.mlir.tpl > /tmp/conv_56.mlir

func.func @conv2d(
    %input:  memref<1xSIZExSIZEx64xf32>,
    %filter: memref<3x3x64x64xf32>,
    %output: memref<1xOSIZExOSIZEx64xf32>) {
  linalg.conv_2d_nhwc_hwcf
    ins(%input, %filter   : memref<1xSIZExSIZEx64xf32>, memref<3x3x64x64xf32>)
    outs(%output          : memref<1xOSIZExOSIZEx64xf32>)
  return
}

func.func @main() {
  %f1 = arith.constant 1.0 : f32
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  %input  = memref.alloc() : memref<1xSIZExSIZEx64xf32>
  %filter = memref.alloc() : memref<3x3x64x64xf32>
  %output = memref.alloc() : memref<1xOSIZExOSIZEx64xf32>

  linalg.fill ins(%f1 : f32) outs(%input  : memref<1xSIZExSIZEx64xf32>)
  linalg.fill ins(%f1 : f32) outs(%filter : memref<3x3x64x64xf32>)
  linalg.fill ins(%f0 : f32) outs(%output : memref<1xOSIZExOSIZEx64xf32>)

  call @conv2d(%input, %filter, %output)
    : (memref<1xSIZExSIZEx64xf32>, memref<3x3x64x64xf32>, memref<1xOSIZExOSIZEx64xf32>) -> ()

  // Expected: 64 * 3 * 3 = 576
  %val = memref.load %output[%c0, %c0, %c0, %c0] : memref<1xOSIZExOSIZEx64xf32>
  vector.print %val : f32

  memref.dealloc %input  : memref<1xSIZExSIZEx64xf32>
  memref.dealloc %filter : memref<3x3x64x64xf32>
  memref.dealloc %output : memref<1xOSIZExOSIZEx64xf32>
  return
}
