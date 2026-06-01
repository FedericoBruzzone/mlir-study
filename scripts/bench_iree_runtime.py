#!/usr/bin/env python3
# Run with: .venv/bin/python3 scripts/bench_iree_runtime.py
#
# Measures per-call IREE latency WITHOUT module-loading overhead.
# Loads each vmfb once, then calls the function 200 times in-process,
# reporting mean ± stddev in ms and estimated GFLOP/s.

import os, sys, time, statistics
import numpy as np
import iree.runtime as iree_rt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
VMFB_DIR    = "/tmp/mlir_iree"
OUT         = os.path.join(RESULTS_DIR, "rq_iree_clean.csv")

MODELS = [
    # (name, vmfb, input_shape, dtype, flops_estimate)
    ("linear_relu_512",    "linear_relu_512.vmfb",    (1, 512),         np.float32, 2*512*512),
    ("linear_relu_1024",   "linear_relu_1024.vmfb",   (1, 1024),        np.float32, 2*1024*1024),
    ("mha_bert_base",      "mha_bert_base.vmfb",      (1, 128, 768),    np.float32, 339_738_624),
    ("mobile_block_56x56", "mobile_block_56x56.vmfb", (1, 32, 56, 56),  np.float32, 28_901_376),
]
# conv_bn_relu_56x56 hits IREE 3.11 compiler crash — excluded

WARMUP = 20
RUNS   = 200

def bench(name, vmfb_path, shape, dtype, flops):
    if not os.path.exists(vmfb_path):
        print(f"  [{name}] vmfb not found — run 'make rq-iree' first")
        return name, float("nan"), float("nan"), float("nan")

    config   = iree_rt.Config("local-task")
    instance = iree_rt.VmInstance()
    vm_mod   = iree_rt.VmModule.mmap(instance, vmfb_path)
    hal_mod  = iree_rt.create_hal_module(instance, config.device)
    ctx      = iree_rt.load_vm_modules(hal_mod, vm_mod, config=config)
    fn  = ctx[-1].main
    dev = config.device

    inp = iree_rt.asdevicearray(dev, np.zeros(shape, dtype=dtype))

    # Warmup
    for _ in range(WARMUP):
        fn(inp)

    # Measure
    times_ms = []
    for _ in range(RUNS):
        t0 = time.perf_counter_ns()
        fn(inp)
        times_ms.append((time.perf_counter_ns() - t0) / 1e6)

    mean_ms   = statistics.mean(times_ms)
    stddev_ms = statistics.stdev(times_ms)
    gflops    = flops / (mean_ms / 1000.0) / 1e9
    return name, mean_ms, stddev_ms, gflops


os.makedirs(RESULTS_DIR, exist_ok=True)
print("=== IREE per-call timing (no module-loading overhead) ===")
print(f"{'model':<25} {'mean_ms':>10} {'stddev_ms':>12} {'GFLOP/s':>10}")

rows = []
for name, vmfb_file, shape, dtype, flops in MODELS:
    vmfb_path = os.path.join(VMFB_DIR, vmfb_file)
    nm, mean, std, gf = bench(name, vmfb_path, shape, dtype, flops)
    rows.append((nm, mean, std, gf))
    print(f"  {nm:<23} {mean:>10.4f} {std:>12.4f} {gf:>10.2f}")

with open(OUT, "w") as f:
    f.write("model,time_mean_ms,time_stddev_ms,gflops\n")
    for nm, mean, std, gf in rows:
        f.write(f"{nm},{mean:.6f},{std:.6f},{gf:.4f}\n")

print(f"\nSaved to {OUT}")
