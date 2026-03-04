# Quantum Plankton Project - Commands & Tasks

## 0. Build Environment (REQUIRED for ARM64/Mac)
docker build --platform linux/amd64 -t quantum-plankton .

## 1. Power Analysis Report (review before running experiments)
docker run --rm --platform linux/amd64 quantum-plankton python utils/power_analysis.py

## Phase 2: Basic Binary Verification
docker run --rm --platform linux/amd64 quantum-plankton python phase2/plankton_ingress.py
docker run --rm --platform linux/amd64 quantum-plankton python phase2/binary_quantum_classifier.py

## Phase 3: Architecture Optimization
docker run --rm --platform linux/amd64 quantum-plankton python phase3/optimize_binary_classifier.py

## Phase 4: Generalized Binary Comparison (25 pairs, ~6 hrs)
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v $(pwd)/phase4/results:/app/phase4/results \
  quantum-plankton python phase4/run_experiments.py

## Phase 4: Smoke Test (~2 min, verifies pipeline only)
docker run --rm --platform linux/amd64 \
  -e SMOKE_TEST=true \
  -v $(pwd)/phase4/results:/app/phase4/results \
  quantum-plankton python phase4/run_experiments.py

## Phase 5: PCA Pipeline Smoke Test (~2 min, verifies PCA + QNN end-to-end)
docker run --rm --platform linux/amd64 \
  quantum-plankton python phase5/smoke_test.py

## Phase 5: K-Category Scaling Benchmarks (k = 2, 3, 4, 5, 8, 12, 16)
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=2.0 \
  -v $(pwd)/phase5/results:/app/phase5/results \
  quantum-plankton python phase5/run_experiments.py

## Phase 5: High-Rigor Scientific Comparison - LOW POWER MODE
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v $(pwd)/phase5/results:/app/phase5/results \
  quantum-plankton python phase5/scientific_comparison.py

## Publish Results (Update READMEs with experiment data)
After running Phase 4 and/or Phase 5 experiments, sync CSV/JSON results
into the Markdown tables in README.md, phase4/README.md, and phase5/README.md:
```bash
python tools/publish_results.py
```
This reads from:
- `phase4/results/experiment_results.csv` → Phase 4 per-pair table
- `phase4/results/aggregate_test.json` → Phase 4 aggregate statistical test
- `phase5/results/comprehensive_k_summary.csv` → Phase 5 K-scaling table

## Test Suite
docker run --rm --platform linux/amd64 quantum-plankton python -m pytest phase4/test_rigor.py -v

## Performance & Thermal Tips
- TF_THREADS=1: Limits CPU to a single core to prevent "flooding".
- EPOCH_COOL=3.0: Pauses for 3s after every training epoch to let the CPU cool.
- THERMAL_SLEEP=60: Pauses for 60s between full experiments.

---
[Back to README](README.md)
