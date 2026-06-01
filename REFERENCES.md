# MLIR References

## Foundational

- **[MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://ieeexplore.ieee.org/document/9370308/)** — Lattner et al., CGO 2021
  The foundational MLIR paper. Start here.

---

## Transform Dialect & Optimization Control

- **[The MLIR Transform Dialect: Your Compiler Is More Powerful Than You Think](https://www.steuwer.info/files/publications/2025/CGO-The-MLIR-Transform-Dialect.pdf)** — CGO 2025
  Declarative, IR-based system for fine-grained control of compiler transformations.

- **[The MLIR Transform Dialect (extended arXiv version)](https://arxiv.org/html/2409.03864v2)** — arXiv 2409.03864, Jan 2026

- **[Transform Dialect Tutorial](https://arxiv.org/pdf/2404.19350)** — arXiv 2404.19350, Apr 2024

- **[Using MLIR Transform to Design Sliced Convolution Algorithm](https://arxiv.org/html/2511.18222v1)** — arXiv 2511.18222, Nov 2025
  SConvTransform: declarative pipeline for 2D convolution optimization via tiling + packing.

---

## AI Compiler & Linalg Pipelines

- **[Towards a High-Performance AI Compiler with Upstream MLIR](https://arxiv.org/html/2404.15204v1)** — arXiv 2404.15204, Apr 2024
  Linalg-on-Tensors-based compilation strategy targeting upstream MLIR.

- **[mlirSynth: Automatic, Retargetable Program Raising](https://arxiv.org/pdf/2310.04196)** — arXiv 2310.04196, 2023
  Automatic lifting of programs to high-level MLIR dialects.

---

## Tiling, Vectorization & Polyhedral

- **[Tiling-Aware Vectorization Framework for Perfect Loop Nests in MLIR](https://link.springer.com/chapter/10.1007/978-981-95-8399-7_30)** — ICA3PP 2025
  Analytical cost model driving tiling + masked vectorization on Linalg. Up to 82% AVX-2 DGEMM peak.

- **[Extending Polygeist to Generate OpenMP SIMD and GPU MLIR Code](https://link.springer.com/chapter/10.1007/978-3-031-90203-1_36)** — Springer 2025
  Polyhedral-optimized (tiling, parallel loops) MLIR code generation.

- **[Progress Report: Deep Learning Guided Exploration of Affine Unimodular Loop Transformations](https://arxiv.org/pdf/2206.03684)** — arXiv 2206.03684, 2022
  DL-based cost model for affine loop transformations; 2.35× speedup over polyhedral compilers.

---

## RL / Automatic Optimization

- **[A Reinforcement Learning Environment for Automatic Code Optimization in the MLIR Compiler](https://arxiv.org/abs/2409.11068)** — CGO 2026
  MLIR-RL: RL agent optimizing Linalg code on CPU from PyTorch/LQCD models. Matches or beats TensorFlow.

---

## Inference & Full-Stack Evaluation

- **[Full-Stack Evaluation of Machine Learning Inference](https://arxiv.org/pdf/2405.15380)** — arXiv 2405.15380, May 2024
  End-to-end evaluation across hardware + compiler stack. Useful as blueprint for empirical study design.

- **[Inference Performance Optimization for Large Language Models on CPUs](https://arxiv.org/pdf/2407.07304)** — arXiv 2407.07304, 2024

---

## IREE (MLIR-based inference runtime)

- **[IREE — official repo & docs](https://iree.dev/)**
  Retargetable MLIR compiler + runtime. CPU backends: `llvm-cpu`, `vmvx`.

- **[TinyIREE: ML Execution Environment for Embedded Systems](https://arxiv.org/pdf/2205.14479)** — arXiv 2205.14479, 2022
  IREE deployment for bare-metal and embedded platforms.

- **[Accelerating GenAI Workloads via RISC-V Microkernel Support in IREE](https://arxiv.org/pdf/2508.14899)** — arXiv 2508.14899, 2025

---

## Hardware & HLS

- **[HIR: An MLIR-based IR for Hardware Accelerator Description](https://dl.acm.org/doi/10.1145/3623278.3624767)** — ASPLOS 2023

- **[Phism: Polyhedral High-Level Synthesis in MLIR](https://arxiv.org/pdf/2103.15103)** — arXiv 2103.15103, 2021

---

## Cost Models

- **[ML-driven Hardware Cost Model for MLIR](https://arxiv.org/pdf/2302.11405)** — arXiv 2302.11405, 2023

---

## Testing & Fuzzing

- **[FLEX: Interleaved Learning and Exploration — Self-Adaptive Fuzz Testing for MLIR](https://arxiv.org/pdf/2510.07815)** — arXiv 2510.07815, 2025
  80 previously unknown bugs found in 30 days; 3.5× more effective than baselines.

---

## Misc / Survey

- **[Enhancing Compiler Design for Machine Learning Workflows with MLIR](https://ijsra.net/content/enhancing-compiler-design-machine-learning-workflows-mlir)** — IJSRA 2025
  Survey on MLIR's role in modern ML compiler stacks.

- **[MLIR-Based Compiler Toolchain (Emergent Mind overview)](https://www.emergentmind.com/topics/mlir-based-compiler-toolchain)**
