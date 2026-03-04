#!/bin/bash
set -euo pipefail

# lightningstart.sh
# Cooldown-sensitive shell script to sequentially run all quickstart.md phases.
# Handles thermal management for CPU-intensive AMD64 emulation on ARM64.
#
# Usage:
#   COOLDOWN_PERIOD=60 ./lightningstart.sh          # default 60s cooldown
#   COOLDOWN_PERIOD=0  ./lightningstart.sh           # no cooldown (CI mode)
#   LOG_DIR=/tmp/logs  ./lightningstart.sh           # custom log directory

# Configuration
COOLDOWN_PERIOD=${COOLDOWN_PERIOD:-60}   # Seconds to wait between phases
IMAGE_NAME="quantum-plankton"
PLATFORM="linux/amd64"
LOG_DIR=${LOG_DIR:-"$(pwd)/logs"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Logging helper: writes to both stdout and log file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "=== Starting Quantum Plankton Pipeline ==="
log "Timestamp: $TIMESTAMP"
log "Cooldown period: $COOLDOWN_PERIOD seconds"
log "Log file: $LOG_FILE"
log "Platform: $PLATFORM"
log ""

PIPELINE_START=$(date +%s)
PHASE_NUM=0

# Run a phase with timing, logging, and cooldown.
# Arguments are passed directly to avoid eval.
run_phase() {
    local label=$1
    shift  # remaining args are the command and its arguments

    PHASE_NUM=$((PHASE_NUM + 1))
    log ""
    log "--- Phase $PHASE_NUM: $label ---"
    log "Command: $*"
    log "Started: $(date '+%Y-%m-%d %H:%M:%S')"

    local phase_start
    phase_start=$(date +%s)

    # Execute the command directly (no eval)
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}

    local phase_end
    phase_end=$(date +%s)
    local elapsed=$((phase_end - phase_start))

    if [ $exit_code -ne 0 ]; then
        log "FAILED: $label (exit code $exit_code, elapsed ${elapsed}s)"
        log "See log: $LOG_FILE"
        exit $exit_code
    fi

    log "Completed: $label (elapsed ${elapsed}s)"

    if [ $COOLDOWN_PERIOD -gt 0 ]; then
        log "Cooling down for $COOLDOWN_PERIOD seconds..."
        sleep $COOLDOWN_PERIOD
    fi
}

# Checksum result artifacts for reproducibility auditing
checksum_results() {
    log ""
    log "--- Result Artifact Checksums ---"
    for dir in phase2/results phase3/results phase4/results phase5/results phase6/results phase7/results; do
        if [ -d "$dir" ]; then
            find "$dir" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.png' \) \
                -exec shasum -a 256 {} \; 2>/dev/null | tee -a "$LOG_FILE"
        fi
    done
}

# Phase 0: Build Environment
# (Tests run during docker build via pytest in the Dockerfile)
run_phase "Build Docker Image" \
    docker build --platform "$PLATFORM" -t "$IMAGE_NAME" .

# Phase 2: Basic Binary Verification (5-fold CV)
run_phase "Data Ingress" \
    docker run --rm --platform "$PLATFORM" \
    "$IMAGE_NAME" python phase2/plankton_ingress.py

run_phase "Binary Classifier (5-fold CV)" \
    docker run --rm --platform "$PLATFORM" \
    -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 \
    -v "$(pwd)/phase2/results:/app/phase2/results" \
    "$IMAGE_NAME" python phase2/binary_quantum_classifier.py

# Phase 3: Architecture Optimization (Nested CV)
run_phase "Hyperparameter Sweep (Nested 5x3 CV)" \
    docker run --rm --platform "$PLATFORM" \
    -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=2.0 -e THERMAL_SLEEP=30 \
    -v "$(pwd)/phase3/results:/app/phase3/results" \
    "$IMAGE_NAME" python phase3/optimize_binary_classifier.py

# Phase 4: Binary Quantum vs. Classical Comparison (Full Run, 5-fold CV)
run_phase "Binary Comparison (5-fold CV, 25 pairs)" \
    docker run --rm --platform "$PLATFORM" \
    -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
    -v "$(pwd)/phase4/results:/app/phase4/results" \
    "$IMAGE_NAME" python phase4/run_experiments.py

# Phase 5a: K-Category Scaling Benchmarks (Full Run)
run_phase "K-Scaling Benchmarks (5-fold CV)" \
    docker run --rm --platform "$PLATFORM" \
    -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=2.0 \
    -v "$(pwd)/phase5/results:/app/phase5/results" \
    "$IMAGE_NAME" python phase5/run_experiments.py

# Phase 5b: Scientific Comparison (K=2,3,4,5 with nested 5x3 CV sweep)
run_phase "Scientific Comparison (nested CV)" \
    docker run --rm --platform "$PLATFORM" \
    -e PYTHONUNBUFFERED=1 -e TF_THREADS=1 -e EPOCH_COOL=3.0 -e THERMAL_SLEEP=60 \
    -v "$(pwd)/phase5/results:/app/phase5/results" \
    "$IMAGE_NAME" python phase5/scientific_comparison.py

# Phase 6: Quantum Saliency Maps
run_phase "Saliency Maps" \
    docker run --rm --platform "$PLATFORM" \
    -v "$(pwd)/phase6/results:/app/phase6/results" \
    "$IMAGE_NAME" python phase6/quantum_saliency.py

# Phase 7: Expressibility & Entanglement Analysis
run_phase "Theoretical Rigor (Expressibility + Entanglement)" \
    docker run --rm --platform "$PLATFORM" \
    -v "$(pwd)/phase7:/app/phase7" \
    "$IMAGE_NAME" python phase7/quantum_rigor.py

# Post-Experiment: Publish Results (runs inside Docker to avoid host dependency)
# Mount project root so publish_results.py can update README.md on the host
run_phase "Publish Results" \
    docker run --rm --platform "$PLATFORM" \
    -v "$(pwd):/app" \
    "$IMAGE_NAME" python tools/publish_results.py

# Artifact integrity
checksum_results

# Pipeline summary
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))
log ""
log "=== All phases completed successfully ==="
log "Total elapsed time: $((TOTAL_ELAPSED / 3600))h $((TOTAL_ELAPSED % 3600 / 60))m $((TOTAL_ELAPSED % 60))s"
log "Log file: $LOG_FILE"
