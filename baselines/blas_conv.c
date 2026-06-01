// Baseline: 2D convolution via im2col + cblas_sgemm (Apple Accelerate).
// Input: 1x58x58x64 NHWC, Filter: 3x3x64x64, Output: 1x56x56x64.
//
// im2col + GEMM is how cuDNN, MKL-DNN and most production frameworks implement
// convolutions on CPUs — makes it an honest baseline.
//
// Build:  clang -O3 -march=native -DACCELERATE_NEW_LAPACK \
//               -isysroot $(xcrun --sdk macosx --show-sdk-path) \
//               blas_conv.c -o blas_conv -framework Accelerate
// Run:    hyperfine --warmup 5 --runs 20 ./blas_conv

#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_IN   1
#define H_IN   58
#define W_IN   58
#define C_IN   64
#define C_OUT  64
#define KH     3
#define KW     3
#define H_OUT  56   // (H_IN - KH + 1) with valid padding
#define W_OUT  56   // (W_IN - KW + 1)

// im2col: flatten each KH×KW×C_IN patch into a row of the col matrix.
// col shape: (H_OUT * W_OUT) × (KH * KW * C_IN)
static void im2col(const float *input, float *col) {
    int row = 0;
    for (int oh = 0; oh < H_OUT; oh++) {
        for (int ow = 0; ow < W_OUT; ow++) {
            int col_idx = 0;
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    // input is NHWC: input[0][ih][iw][:]
                    const float *src = input + (ih * W_IN + iw) * C_IN;
                    memcpy(col + row * (KH * KW * C_IN) + col_idx,
                           src, C_IN * sizeof(float));
                    col_idx += C_IN;
                }
            }
            row++;
        }
    }
}

static void fill_random(float *buf, size_t n) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

int main(void) {
    srand(42);

    const int M = H_OUT * W_OUT;          // output spatial positions
    const int K = KH * KW * C_IN;         // patch size
    const int N = C_OUT;                   // output channels

    float *input  = aligned_alloc(64, H_IN * W_IN * C_IN * sizeof(float));
    float *filter = aligned_alloc(64, K * N * sizeof(float)); // K × C_OUT
    float *col    = aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *output = aligned_alloc(64, (size_t)M * N * sizeof(float));
    if (!input || !filter || !col || !output) {
        fputs("alloc failed\n", stderr); return 1;
    }

    fill_random(input,  H_IN * W_IN * C_IN);
    fill_random(filter, K * N);
    memset(output, 0, (size_t)M * N * sizeof(float));

    im2col(input, col);

    // GEMM: output(M×N) = col(M×K) × filter(K×N)
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, col, K,
                      filter, N,
                0.0f, output, N);

    printf("%.6f\n", (double)output[0]);

    free(input); free(filter); free(col); free(output);
    return 0;
}
