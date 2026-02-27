# Quantum Plankton Project - Commands & Tasks

## 0. Build Environment (REQUIRED for ARM64/Mac)
docker build --platform linux/amd64 -t quantum-plankton .

## Phase 2: Basic Binary Verification
docker run --rm --platform linux/amd64 quantum-plankton python phase2/plankton_ingress.py
docker run --rm --platform linux/amd64 quantum-plankton python phase2/binary_quantum_classifier.py

## Phase 3: Architecture Optimization
docker run --rm --platform linux/amd64 quantum-plankton python phase3/optimize_binary_classifier.py

## Phase 4: Generalized Binary Comparison (4x4 PCA) - LOW POWER MODE
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v $(pwd)/phase4/results:/app/phase4/results \
  quantum-plankton python phase4/run_experiments.py

## Phase 5: K-Category Scaling Study (4x4 PCA) - LOW POWER MODE
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v $(pwd)/phase5/results:/app/phase5/results \
  quantum-plankton python phase5/run_experiments.py

## Phase 5: High-Rigor Scientific Comparison - LOW POWER MODE
docker run -it --rm --platform linux/amd64 \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v $(pwd)/phase5/results:/app/phase5/results \
  quantum-plankton python phase5/scientific_comparison.py

## Performance & Thermal Tips
- TF_THREADS=1: Limits CPU to a single core to prevent "flooding".
- EPOCH_COOL=3.0: Pauses for 3s after every training epoch to let the CPU cool.
- THERMAL_SLEEP=60: Pauses for 60s between full experiments.
\n---\n[Back to README](README.md)
