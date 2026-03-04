#!/bin/bash

# lightningstart.sh
# Cooldown-sensitive shell script to sequentially run all quickstart.md phases.
# Handles thermal management for CPU-intensive AMD64 emulation on ARM64.

# Configuration
COOLDOWN_PERIOD=${COOLDOWN_PERIOD:-60} # Seconds to wait between phases
IMAGE_NAME="quantum-plankton"
PLATFORM="linux/amd64"

echo "=== Starting Quantum Plankton Pipeline ==="
echo "Cooldown period: $COOLDOWN_PERIOD seconds"

# Function to run a command with cooldown
run_phase() {
    local label=$1
    local cmd=$2
    echo ""
    echo "--- Phase: $label ---"
    echo "Running: $cmd"
    
    # Execute the command
    eval "$cmd"
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Error: $label failed with exit code $exit_code"
        exit $exit_code
    fi
    
    echo "Completed: $label"
    echo "Cooling down for $COOLDOWN_PERIOD seconds..."
    sleep $COOLDOWN_PERIOD
}

# 0. Build Environment
run_phase "Build Docker Image" "docker build --platform $PLATFORM -t $IMAGE_NAME ."

# 2. Basic Binary Verification
run_phase "Phase 2: Data Ingress" "docker run --rm --platform $PLATFORM $IMAGE_NAME python phase2/plankton_ingress.py"
run_phase "Phase 2: Binary Classifier" "docker run --rm --platform $PLATFORM $IMAGE_NAME python phase2/binary_quantum_classifier.py"

# 3. Architecture Optimization
run_phase "Phase 3: Hyperparameter Sweep" "docker run --rm --platform $PLATFORM $IMAGE_NAME python phase3/optimize_binary_classifier.py"

# 4. Binary Quantum vs. Classical Comparison (Full Run)
run_phase "Phase 4: Full Experiments" "docker run -it --rm --platform $PLATFORM \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v \$(pwd)/phase4/results:/app/phase4/results \
  $IMAGE_NAME python phase4/run_experiments.py"

# 5. K-Category Scaling Benchmarks (Full Run)
run_phase "Phase 5: Full Experiments" "docker run -it --rm --platform $PLATFORM \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=2.0 \
  -v \$(pwd)/phase5/results:/app/phase5/results \
  $IMAGE_NAME python phase5/run_experiments.py"

# 5. Scientific Comparison (K=2,3,4,5 with sweep)
run_phase "Phase 5: Scientific Comparison" "docker run -it --rm --platform $PLATFORM \
  -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
  -v \$(pwd)/phase5/results:/app/phase5/results \
  $IMAGE_NAME python phase5/scientific_comparison.py"

# 6. Quantum Saliency Maps
run_phase "Phase 6: Saliency Maps" "docker run --rm --platform $PLATFORM -v \$(pwd)/phase6/results:/app/phase6/results $IMAGE_NAME python phase6/quantum_saliency.py"

# 7. Expressibility & Entanglement Analysis
run_phase "Phase 7: Theoretical Rigor" "docker run --rm --platform $PLATFORM -v \$(pwd)/phase7:/app/phase7 $IMAGE_NAME python phase7/quantum_rigor.py"

# Post-Experiment: Publish Results
run_phase "Publish Results" "python tools/publish_results.py"

echo ""
echo "=== All phases completed successfully ==="
