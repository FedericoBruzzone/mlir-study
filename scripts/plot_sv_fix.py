#!/usr/bin/env python3
# Plot: sv-fix patch benchmark results
# Usage: .venv/bin/python3 scripts/plot_sv_fix.py

import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

rows = []
with open(os.path.join(RESULTS, "sv_fix.csv")) as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["time_mean_s"] = float(r["time_mean_s"])
        r["time_stddev_s"] = float(r["time_stddev_s"])
        rows.append(r)

sizes = sorted(set(r["size_N"] for r in rows), key=int)
paths_ordered = ["affine_tiled", "vec_gather", "vec_interchange", "vec_inbounds"]
colors = {"affine_tiled": "#2ecc71", "vec_gather": "#e74c3c",
          "vec_interchange": "#f39c12", "vec_inbounds": "#3498db"}
labels = {"affine_tiled": "A: affine_tiled (scalar)",
          "vec_gather": "B: vec_gather (k-loop)",
          "vec_interchange": "C: vec_interchange (masked)",
          "vec_inbounds": "D: vec_inbounds (fix)"}

# ── Figure 1: grouped bar chart, time per matmul ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(sizes))
w = 0.2

for i, p in enumerate(paths_ordered):
    times = []
    errs = []
    for s in sizes:
        r = [r for r in rows if r["path"] == p and r["size_N"] == s][0]
        times.append(r["time_mean_s"])
        errs.append(r["time_stddev_s"])
    offset = (i - 1.5) * w
    ax.bar(x + offset, times, w, yerr=errs, capsize=3,
           label=labels[p], color=colors[p], edgecolor="white", linewidth=0.5)

ax.set_xlabel("Matrix size N", fontsize=12)
ax.set_ylabel("Time per matmul (s)", fontsize=12)
ax.set_title("affine-super-vectorize: in_bounds fix benchmark (M4, T=16)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"N={s}" for s in sizes], fontsize=11)
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "sv_fix.png"), dpi=150)
print(f"Saved {RESULTS}/sv_fix.png")

# ── Figure 2: speedup relative to scalar baseline ───────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
for i, p in enumerate(paths_ordered):
    ratios = []
    for s in sizes:
        r = [r for r in rows if r["path"] == p and r["size_N"] == s][0]
        r_base = [r for r in rows if r["path"] == "affine_tiled" and r["size_N"] == s][0]
        ratios.append(r["time_mean_s"] / r_base["time_mean_s"])
    offset = (i - 1.5) * w
    ax2.bar(x + offset, ratios, w, label=labels[p], color=colors[p],
            edgecolor="white", linewidth=0.5)

ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="scalar baseline")
ax2.set_xlabel("Matrix size N", fontsize=12)
ax2.set_ylabel("Slowdown vs scalar (×)", fontsize=12)
ax2.set_title("Slowdown relative to Path A (affine_tiled scalar)", fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels([f"N={s}" for s in sizes], fontsize=11)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "sv_fix_slowdown.png"), dpi=150)
print(f"Saved {RESULTS}/sv_fix_slowdown.png")
