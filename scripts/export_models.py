#!/usr/bin/env python3
# Run with: .venv/bin/python3 scripts/export_models.py  (or via make models)
"""
Export real PyTorch model layers to MLIR via iree-turbine.
Outputs Linalg-level MLIR files into kernels/models/.

Usage: python3 scripts/export_models.py
Requires: pip install iree-turbine torch  (see venv/requirements.txt)
"""

import os, sys
import torch
import iree.turbine.aot as aot

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "kernels", "models")
os.makedirs(OUT_DIR, exist_ok=True)


def export(name: str, module: torch.nn.Module, *example_args):
    exported = aot.export(module, args=example_args)
    path = os.path.join(OUT_DIR, f"{name}.mlir")
    with open(path, "w") as f:
        f.write(str(exported.mlir_module))
    print(f"[OK] {name} → {path}")


# ── 1. Linear layer (512→512) — represents a transformer FFN projection ──────
class LinearReLU(torch.nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.fc = torch.nn.Linear(d, d, bias=True)
    def forward(self, x): return torch.relu(self.fc(x))

export("linear_relu_512",   LinearReLU(512),  torch.randn(1, 512))
export("linear_relu_1024",  LinearReLU(1024), torch.randn(1, 1024))

# ── 2. Conv2d layer — ResNet-50 first residual block (56×56, 64→64, 3×3) ────
class ConvBnReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn   = torch.nn.BatchNorm2d(64)
    def forward(self, x): return torch.relu(self.bn(self.conv(x)))

export("conv_bn_relu_56x56", ConvBnReLU(), torch.randn(1, 64, 56, 56))

# ── 3. Multi-head attention (BERT-base: 12 heads, d_model=768, seq=128) ──────
class MHAttention(torch.nn.Module):
    def __init__(self, d=768, heads=12, seq=128):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d, heads, batch_first=True)
    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return out

export("mha_bert_base", MHAttention(), torch.randn(1, 128, 768))

# ── 4. Depthwise + pointwise (MobileNetV2-style block, 32→16, 3×3) ──────────
class MobileBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.pw = torch.nn.Conv2d(32, 16, 1, bias=False)
    def forward(self, x): return self.pw(torch.nn.functional.relu6(self.dw(x)))

export("mobile_block_56x56", MobileBlock(), torch.randn(1, 32, 56, 56))
