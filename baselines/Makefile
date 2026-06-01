CC     := $(shell xcrun --find clang)
SDK    := $(shell xcrun --sdk macosx --show-sdk-path)
CFLAGS := -O3 -march=native -isysroot $(SDK) -framework Accelerate

SIZES  := 128 256 512 1024

BLAS_MATMUL_BINS := $(patsubst %, blas_matmul_%, $(SIZES))

.PHONY: all clean matmul conv

all: matmul conv

matmul: $(BLAS_MATMUL_BINS)

blas_matmul_%: blas_matmul.c
	$(CC) $(CFLAGS) -DSIZE=$* $< -o $@

conv: blas_conv
blas_conv: blas_conv.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f blas_matmul_* blas_conv
