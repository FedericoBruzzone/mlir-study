// Baseline: NxN FP32 matmul via Apple Accelerate (cblas_sgemm).
// Represents the "ceiling" — highly optimised for M-series (uses AMX coprocessor).
//
// Build:  clang -O3 -march=native -DSIZE=512 blas_matmul.c -o blas_matmul_512 \
//               -framework Accelerate
// Run:    ./blas_matmul_512 [NITER]   (default NITER=1)
//   NITER>1 amortises process-startup overhead: report hyperfine wall-time / NITER.

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SIZE
#define SIZE 512
#endif

static void fill_random(float *buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

int main(int argc, char *argv[]) {
    int niter = argc > 1 ? atoi(argv[1]) : 1;
    srand(42);

    float *A = aligned_alloc(64, (size_t)SIZE * SIZE * sizeof(float));
    float *B = aligned_alloc(64, (size_t)SIZE * SIZE * sizeof(float));
    float *C = aligned_alloc(64, (size_t)SIZE * SIZE * sizeof(float));
    if (!A || !B || !C) { fputs("alloc failed\n", stderr); return 1; }

    fill_random(A, SIZE * SIZE);
    fill_random(B, SIZE * SIZE);
    memset(C, 0, (size_t)SIZE * SIZE * sizeof(float));

    for (int iter = 0; iter < niter; iter++) {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    SIZE, SIZE, SIZE,
                    1.0f, A, SIZE,
                          B, SIZE,
                    0.0f, C, SIZE);
    }

    // Anti-optimisation guard: print one value so the compiler cannot elide the call
    printf("%.6f\n", (double)C[0]);

    free(A); free(B); free(C);
    return 0;
}
