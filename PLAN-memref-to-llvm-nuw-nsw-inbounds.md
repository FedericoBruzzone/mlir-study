# Plan: Add `nuw nsw inbounds` to memref→LLVM address arithmetic

## Problem

When `affine-super-vectorize` emits `vector.transfer_read/write` with
`in_bounds = [true]`, the lowering to LLVM IR produces address arithmetic
(`mul`/`add`/`getelementptr`) **without** `nuw`/`nsw`/`inbounds` flags.

Without these flags, LLVM's Scalar Evolution (SCEV) cannot prove that the
loop-internal store to C (a reduction accumulator) doesn't alias with other
accesses. Result: **C is stored-to-memory and reloaded every iteration**
instead of being accumulated in a NEON register. On M4 at N=512, this costs
~21% overhead (108ms vs 90ms for the scalar LLVM auto-vectorized path).

## Root cause

When `--finalize-memref-to-llvm` (and the affine→LLVM lowering) translates
memref descriptor arithmetic into LLVM IR, it emits plain `mul`/`add`/`GEP`
without overflow flags.

Contrast with LLVM's **own Loop Vectorizer**: it starts from scalar
`load`/`store` + `getelementptr inbounds` (emitted by Clang from C/C++).
SCEV sees `inbounds` and `nuw` → proves loop bounds → identifies reduction
pattern → accumulates in register → single store at loop exit.

## Evidence

**Path A** (scalar loop → LLVM auto-vectorize → fast):
```llvm
%mul = mul nuw nsw i64 %i, 512
%add = add nuw nsw i64 %mul, %j
%gep = getelementptr inbounds nuw float, ptr %base, i64 %add
```

**Path D** (MLIR `vector.transfer_read/write` → LLVM → slow):
```llvm
%mul = mul i64 %i, 512        ; ← no nuw nsw
%add = add i64 %mul, %j       ; ← no nuw nsw
%gep = getelementptr float, ptr %base, i64 %add  ; ← no inbounds
```

## Missing metadata (Path A vs Path D)

| Annotation           | Path A | Path D |
|----------------------|--------|--------|
| `mul nuw nsw`        | ✅     | ❌     |
| `add nuw nsw`        | ✅     | ❌     |
| `getelementptr inbounds` | ✅ | ❌     |
| `llvm.access.group`  | ✅     | ❌     |
| `alias.scope`        | ✅     | ❌     |
| `noalias`            | ✅     | ❌     |

## Fix strategy

### Option A — Add flags in `FinalizeMemrefToLLVM` (recommended)

**File**: `mlir/lib/Conversion/MemRefToLLVM/FinalizeMemrefToLLVM.cpp`

In the address-computation helpers (`computeAlignment`, `getStridedElementPtr`,
or equivalent), add `nuw nsw` to `llvm.add`/`llvm.mul` ops and `inbounds` to
`llvm.getelementptr` when the memref dimension is statically known and the index
arithmetic is proven non-wrapping.

**Key insight**: memref indices are always non-negative (`index` type is
unsigned in practice) and the GEP offset is within the allocated size
(statically checked or dynamically guarded). Adding `inbounds` is correct
when the memref access is `in_bounds = [true]` or when the memref
dimensions are static and the loop bounds don't exceed them.

**Relevant code locations** (LLVM trunk, circa June 2025):
- `mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp` — contains
  `getStridedElementPtr` which generates the GEP
- `mlir/lib/Conversion/MemRefToLLVM/AllocLikeConversion.cpp` — alloc lowering
- `mlir/lib/Dialect/Vector/Transforms/LowerVectorTransfer.cpp` — lower
  `vector.transfer_read/write` to `vector.load/store`

### Option B — Add `inbounds` in vector-transfer lowering only

Target the specific issue by adding `inbounds` to the GEP emitted when
lowering `vector.transfer_read/write` with `in_bounds = [true]`.

This is narrower-scope but doesn't fix the general memref lowering.

### Option C — Add `llvm.access.group` / alias metadata

In addition to `inbounds`, add `llvm.access.group` metadata to help LLVM
identify loop nests and apply store-to-load forwarding / reduction
accumulation. This is what LLVM's own Loop Vectorizer does via
`llvm.metadata` attached to `load`/`store` instructions.

## Expected impact

| Metric | Before | After |
|--------|--------|-------|
| C store+reload per iteration | 12× (unrolled) | 0× (register) |
| tbz (masked branch) | 0 (already fixed) | 0 |
| `fmla` (fused multiply-add) | 0 | 12× (replace fmul+fadd pair) |
| Time vs scalar baseline | 1.21× | ~1.0× |

## Validation

1. **LLVM IR check**: grep for `inbounds` on all GEPs in lowered code
2. **Assembly check**: grep for `str q` inside innermost loop body → should be 0
3. **Assembly check**: grep for `fmla` → should match unroll factor
4. **Benchmark**: `bash scripts/bench_sv_fix.sh` → D matches A within noise

## Notes

- This is an LLVM-middleware issue, not an MLIR-algorithm issue. The
  lowering pipeline discards semantic information (non-wrapping indices,
  non-aliasing memrefs) that LLVM needs.
- The `affine` dialect has the information (loop bounds are static, memref
  sizes are known) but it's lost during lowering to LLVM.
- A cleaner long-term fix: teach the LLVM `inbounds` propagation pass to
  infer `nuw nsw` from loop structure, independent of MLIR.
