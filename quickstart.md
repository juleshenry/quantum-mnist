# Quantum Plankton Project - Quick Start Guide

This guide walks through the complete experiment pipeline (Phases 1-7) end to end.

## 0. Build Environment (REQUIRED)

Build the Docker image. This pins all dependencies and runs the automated test
suite (`test_rigor.py`). The build **aborts** if any test fails.

```bash
docker build --platform linux/amd64 -t quantum-plankton .
```

> **ARM64/Mac users:** The `--platform linux/amd64` flag is required for
> TensorFlow-Quantum compatibility. See "Performance & Thermal Tips" at the end
> for heat-management flags.

---

## Full Phase Suite

### Phase 1: Confirm Original Research (Notebook)

Phase 1 is the Jupyter notebook `quantum_image_mnist.ipynb`. Run it in Google
Colab or locally to confirm the original MNIST quantum classification results.

### Phase 2: Basic Binary Verification

Verify data ingress and run a basic binary quantum classifier on plankton data.

```bash
docker run --rm --platform linux/amd64 \
  quantum-plankton python phase2/plankton_ingress.py

docker run --rm --platform linux/amd64 \
  quantum-plankton python phase2/binary_quantum_classifier.py
```

### Phase 3: Architecture Optimization (Hyperparameter Sweep)

Run a multi-trial random sweep over encoding strategies, circuit depth, and
optimization parameters.

```bash
docker run --rm --platform linux/amd64 \
  quantum-plankton python phase3/optimize_binary_classifier.py
```

### Phase 4: Binary Quantum vs. Classical Comparison (~6 hrs)

High-rigor comparison: 5-fold CV across 25 plankton pairs with paired t-tests
and Holm-Bonferroni correction.

```bash
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v $(pwd)/phase4/results:/app/phase4/results \
  quantum-plankton python phase4/run_experiments.py
```

**Smoke test** (~2 min, verifies pipeline only):

```bash
docker run --rm --platform linux/amd64 \
  -e SMOKE_TEST=true \
  -v $(pwd)/phase4/results:/app/phase4/results \
  quantum-plankton python phase4/run_experiments.py
```

**Outputs:**
- `phase4/results/experiment_results.csv` -- per-fold, per-pair metrics
- `phase4/results/experiment_summary.csv` -- aggregated stats with p-values
- `phase4/results/aggregate_test.json` -- aggregate QNN vs. Fair test
- `phase4/results/experiment_config.json` -- full experiment config
- `phase4/results/confusion_matrices/` -- per-fold confusion matrices

### Phase 5: K-Category Scaling Benchmarks

Multi-class scaling study (k = 2, 3, 4, 5, 8, 12, 16) with PCA pipeline.

**Smoke test** (~2 min, verifies PCA + QNN end-to-end):

```bash
docker run --rm --platform linux/amd64 \
  quantum-plankton python phase5/smoke_test.py
```

**Full run** (k = 2..16):

```bash
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=2.0 \
  -v $(pwd)/phase5/results:/app/phase5/results \
  quantum-plankton python phase5/run_experiments.py
```

**Scientific comparison** (K=2,3,4,5 with hyperparameter sweep):

```bash
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v $(pwd)/phase5/results:/app/phase5/results \
  quantum-plankton python phase5/scientific_comparison.py
```

**Outputs:**
- `phase5/results/comprehensive_k_results.csv` -- per-fold, per-K metrics
- `phase5/results/comprehensive_k_summary.csv` -- aggregated with p-values
- `phase5/results/k_scaling_comparison.png` -- accuracy/F1 plots
- `phase5/results/scientific_k_comparison.csv` -- swept comparison metrics
- `phase5/results/scientific_k_summary.csv` -- swept summary with p-values
- `phase5/results/scientific_scaling_plot.png` -- scaling plot with significance markers

### Phase 6: Quantum Saliency Maps

Generate gradient-based saliency maps showing which pixels drive QNN decisions.

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/phase6/results:/app/phase6/results \
  quantum-plankton python phase6/quantum_saliency.py
```

**Outputs:**
- `phase6/results/saliency_example_0.png` through `saliency_example_4.png` -- original image, heatmap, and overlay for each example

### Phase 7: Expressibility & Entanglement Analysis

Theoretical rigor analysis of the PQC architecture using Meyer-Wallach
entanglement and KL divergence expressibility metrics across 1-5 layers.

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/phase7:/app/phase7 \
  quantum-plankton python phase7/quantum_rigor.py
```

**Outputs:**
- `phase7/results_rigor.txt` -- expressibility and entanglement metrics per layer count

---

## Post-Experiment: Publish Results

After running Phase 4 and/or Phase 5 experiments, sync CSV/JSON results into
the Markdown tables in `README.md`, `phase4/README.md`, and `phase5/README.md`:

```bash
python tools/publish_results.py
```

This reads from:
- `phase4/results/experiment_results.csv` -- Phase 4 per-pair table
- `phase4/results/aggregate_test.json` -- Phase 4 aggregate statistical test
- `phase5/results/comprehensive_k_summary.csv` -- Phase 5 K-scaling table

---

## Utilities

### Power Analysis Report

Review the statistical justification for pair count and selection before
running experiments:

```bash
docker run --rm --platform linux/amd64 \
  quantum-plankton python utils/power_analysis.py
```

### Test Suite

Run the verification test suite standalone:

```bash
docker run --rm --platform linux/amd64 \
  quantum-plankton python -m pytest phase4/test_rigor.py -v
```

---

## Performance & Thermal Tips

When running on ARM64 Macs (M1/M2/M3), AMD64 emulation is CPU-intensive. The
following environment variables help manage heat:

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_THREADS` | `1` | Limits CPU to a single core |
| `EPOCH_COOL` | `1.0` | Seconds to pause after every training epoch |
| `THERMAL_SLEEP` | `0` | Seconds to pause between full experiments |
| `BATCH_COOL` | `0` | Seconds to pause after every batch |
| `BREATHE_SLEEP` | `0.05` | Micro-sleeps during heavy data processing |

**Maximum cooling one-liner:**

```bash
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 \
  -e EPOCH_COOL=5.0 -e THERMAL_SLEEP=120 -e BREATHE_SLEEP=0.2 \
  -v $(pwd)/phase4/results:/app/phase4/results \
  quantum-plankton python phase4/run_experiments.py
```

### Customizable Experiment Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `N_FOLDS` | `5` | Number of cross-validation folds |
| `Q_SAMPLES` | `200` (P4) / `400` (P5) | Max training samples per model |
| `SMOKE_TEST` | `false` | Reduce to 1 pair/K, 2 folds, 10 samples |
| `DATA_DIR` | `/app/data/zooplankton_0p5x` | Path to plankton dataset |
| `RESULTS_DIR` | `phaseN/results` | Output directory |

---
[Back to README](README.md)
