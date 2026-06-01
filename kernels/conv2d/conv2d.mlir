// 2D convolution — NHWC layout (N=1, H=58, W=58, C_in=64).
// Filter: 3x3x64x64 (kH x kW x C_in x C_out).
// Output: 1x56x56x64 (valid padding, no stride).
//
// Sizes mirror ResNet-50 layer 1 (56x56 feature maps, 64 channels).
// Expected: output[0][0][0][0] = sum of 3*3*64 = 576 products.

func.func @conv2d(
    %input:  memref<1x58x58x64xf32>,
    %filter: memref<3x3x64x64xf32>,
    %output: memref<1x56x56x64xf32>) {
  linalg.conv_2d_nhwc_hwcf
    ins(%input, %filter   : memref<1x58x58x64xf32>, memref<3x3x64x64xf32>)
    outs(%output          : memref<1x56x56x64xf32>)
  return
}

func.func @main() {
  %f1 = arith.constant 1.0 : f32
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  %input  = memref.alloc() : memref<1x58x58x64xf32>
  %filter = memref.alloc() : memref<3x3x64x64xf32>
  %output = memref.alloc() : memref<1x56x56x64xf32>

  linalg.fill ins(%f1 : f32) outs(%input  : memref<1x58x58x64xf32>)
  linalg.fill ins(%f1 : f32) outs(%filter : memref<3x3x64x64xf32>)
  linalg.fill ins(%f0 : f32) outs(%output : memref<1x56x56x64xf32>)

  call @conv2d(%input, %filter, %output)
    : (memref<1x58x58x64xf32>, memref<3x3x64x64xf32>, memref<1x56x56x64xf32>) -> ()

  // C_in * kH * kW = 64 * 3 * 3 = 576
  %val = memref.load %output[%c0, %c0, %c0, %c0] : memref<1x56x56x64xf32>
  vector.print %val : f32

  memref.dealloc %input  : memref<1x58x58x64xf32>
  memref.dealloc %filter : memref<3x3x64x64xf32>
  memref.dealloc %output : memref<1x56x56x64xf32>
  return
}
