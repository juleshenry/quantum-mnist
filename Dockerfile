# Comprehensive Dockerfile for Quantum Plankton Project (All Phases)
FROM tensorflow/tensorflow:2.7.0

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies - specific versions for TFQ compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow-quantum==0.7.2 \
    cirq==0.13.1 \
    sympy==1.8 \
    pillow==9.0.1 \
    numpy==1.21.6 \
    pandas==1.3.5 \
    matplotlib==3.5.1 \
    seaborn==0.11.2 \
    tqdm==4.64.0 \
    scikit-learn==1.0.2 \
    pytest==7.1.2

WORKDIR /app

# Copy the entire project
COPY . .

# Environment setup
ENV DATA_DIR="/app/data/zooplankton_0p5x"
ENV PYTHONPATH="/app:/app/utils:/app/phase1:/app/phase2:/app/phase3:/app/phase4:/app/phase5:/app/phase6:/app/phase7"

# Ensure results directories exist
RUN mkdir -p /app/phase2/results /app/phase3/results /app/phase4/results /app/phase5/results /app/phase6/results /app/phase7/results /app/utils

# Run the comprehensive rigor test suites before experiments.
# If any test fails, the build aborts.
RUN python -m pytest \
    /app/phase2/test_rigor_phase2.py \
    /app/phase3/test_rigor_phase3.py \
    /app/phase4/test_rigor.py \
    /app/phase5/test_rigor_phase5.py \
    /app/phase6/test_rigor_phase6.py \
    /app/phase7/test_rigor_phase7.py \
    -v --tb=short

# Default to running the latest phase (Phase 5)
CMD ["python", "phase5/run_experiments.py"]
