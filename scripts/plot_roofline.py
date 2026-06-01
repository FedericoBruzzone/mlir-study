#!/usr/bin/env python3
# Run with: .venv/bin/python3 scripts/plot_roofline.py
#
# Generates results/roofline.pdf — the central figure of the paper.
# Roofline model for Apple M4 Pro (single performance core, FP32).

import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Hardware constants (M4 Pro, single perf core) ─────────────────────────
PEAK_SCALAR   = 9.0    # GFLOP/s  (2 FLOPs/cycle × 4.5 GHz, no SIMD)
PEAK_NEON     = 36.0   # GFLOP/s  (8 f32 lanes × 2 FLOPs/cycle × 4.5 GHz)
PEAK_AMX      = 182.0  # GFLOP/s  (Accelerate BLAS measured at N=1024 — proxy for AMX peak)
MEM_BW        = 68.0   # GB/s

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Data points  (name, arithmetic_intensity FLOPs/B, measured GFLOP/s, marker) ─
DATA = []

# Matmul: AI = 2N³ / (3N²×4B) = N/6
def matmul_ai(n): return n / 6.0

# Load RQ5 for MLIR points
rq5 = {}
rq5_file = os.path.join(RESULTS, "rq5_vs_baseline.csv")
if os.path.exists(rq5_file):
    with open(rq5_file) as f:
        for row in csv.DictReader(f):
            key = (int(row["size_N"]), row["variant"])
            if row["gflops"] not in ("N/A", ""):
                rq5[key] = float(row["gflops"])

for N in [128, 256, 512, 1024]:
    ai = matmul_ai(N)
    if (N, "mlir_affine_t16")  in rq5: DATA.append((f"MLIR tiled T=16\nN={N}",  ai, rq5[(N,"mlir_affine_t16")],  "o", "#2196F3"))
    if (N, "mlir_vector_t16")  in rq5: DATA.append((f"MLIR+NEON T=16\nN={N}",   ai, rq5[(N,"mlir_vector_t16")],  "^", "#4CAF50"))
    if (N, "accelerate_blas")  in rq5: DATA.append((f"Accelerate\nN={N}",        ai, rq5[(N,"accelerate_blas")],  "D", "#F44336"))

# Load IREE clean if available
# Arithmetic intensities (FLOPs / bytes_accessed):
#   linear 512:  AI = 2*512*512 / (2*512*512*4) = 0.25 FLOPs/B (very low: weight-dominated)
#   mha_bert:    AI ≈ 16 FLOPs/B (large QKV matmuls dominate)
#   mobile:      AI ≈ 9 FLOPs/B  (depthwise conv, compact filters)
IREE_AI = {
    "linear_relu_512":    0.25,
    "linear_relu_1024":   0.25,
    "mha_bert_base":     16.0,
    "mobile_block_56x56": 9.0,
}
iree_file = os.path.join(RESULTS, "rq_iree_clean.csv")
if os.path.exists(iree_file):
    with open(iree_file) as f:
        for row in csv.DictReader(f):
            if row["gflops"] not in ("nan", "N/A", ""):
                gf  = float(row["gflops"])
                ai  = IREE_AI.get(row["model"], 1.0)
                lbl = "IREE " + row["model"].replace("_", " ")
                DATA.append((lbl, ai, gf, "s", "#FF9800"))

# Softmax from RQ4 (AI ≈ 0.63 FLOPs/B)
rq4_file = os.path.join(RESULTS, "rq4_workloads.csv")
if os.path.exists(rq4_file):
    with open(rq4_file) as f:
        for row in csv.DictReader(f):
            if row["kernel"] == "softmax" and row["variant"] == "scf":
                t = float(row["time_mean_s"])
                n = 512
                flops = 5.0 * n * n
                gf = flops / (t * 1e9)
                DATA.append(("Softmax\n512×512", 0.63, gf, "P", "#9C27B0"))

# Conv2d: AI roughly 16 FLOPs/B (compute-bound)
if os.path.exists(rq4_file):
    with open(rq4_file) as f:
        for row in csv.DictReader(f):
            if row["kernel"] == "conv2d" and row["variant"] == "affine_tiled_16":
                t = float(row["time_mean_s"])
                flops = 2 * 56*56 * 64 * 3*3*64
                gf = flops / (t * 1e9)
                DATA.append(("Conv2d\n56×56×64\ntiled T=16", 16.0, gf, "h", "#00BCD4"))

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)

# X range
ai_range = np.logspace(-2, 9, 500, base=2)

# Roofline lines
def roof(ai_arr, peak_gflops, bw_gbs):
    return np.minimum(ai_arr * bw_gbs, peak_gflops)

ax.plot(ai_range, roof(ai_range, PEAK_SCALAR, MEM_BW),
        "--", color="gray",    lw=1.4, label=f"Scalar peak ({PEAK_SCALAR} GFLOP/s)")
ax.plot(ai_range, roof(ai_range, PEAK_NEON,   MEM_BW),
        "-",  color="#FF9800", lw=2.0, label=f"NEON peak ({PEAK_NEON} GFLOP/s)")
ax.plot(ai_range, roof(ai_range, PEAK_AMX,    MEM_BW),
        "-",  color="#F44336", lw=2.0, label=f"Accelerate/AMX ({PEAK_AMX} GFLOP/s)")

# Bandwidth roof label
ax.text(0.9, 1.6 * MEM_BW * 0.7, f"Mem BW {MEM_BW} GB/s",
        rotation=25, fontsize=8, color="steelblue", alpha=0.8)

# Ridge points
for peak, col in [(PEAK_NEON, "#FF9800"), (PEAK_AMX, "#F44336")]:
    ridge = peak / MEM_BW
    ax.axvline(ridge, color=col, lw=0.8, linestyle=":", alpha=0.5)

# Data points
seen_labels = {}
for (name, ai, gf, mk, col) in DATA:
    base = name.split("\n")[0]
    lbl  = base if base not in seen_labels else None
    seen_labels[base] = True
    # Softmax gets a larger marker so it's visible at low GFLOP/s
    size = 160 if "Softmax" in name else 90
    ax.scatter(ai, gf, marker=mk, color=col, s=size, zorder=5, label=lbl,
               edgecolors="white", linewidths=0.5)
    short = name.replace("\n", " ").split("N=")[-1] if "N=" in name else name.split("\n")[0]
    ax.annotate(short, (ai, gf), textcoords="offset points",
                xytext=(6, 4), fontsize=6.5, color=col)

# Axes
ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=11)
ax.set_ylabel("Performance (GFLOP/s)", fontsize=11)
ax.set_title("Roofline — Apple M4 Pro (single thread, FP32)\nMLIR naive vs NEON vs Accelerate", fontsize=11)
ax.set_xlim(2**-2, 2**9)
ax.set_ylim(2**-8, 2**9)   # extended down to show softmax (~0.03 GFLOP/s ≈ 2^-5)
ax.grid(True, which="both", alpha=0.2)
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
out = os.path.join(RESULTS, "roofline.pdf")
plt.savefig(out, dpi=200)
print(f"Saved: {out}")

# Also save PNG for quick preview
png = out.replace(".pdf", ".png")
plt.savefig(png, dpi=150)
print(f"Saved: {png}")
