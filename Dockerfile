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
    scikit-learn==1.0.2

WORKDIR /app

# Copy the entire project
COPY . .

# Environment setup
ENV DATA_DIR="/app/data/zooplankton_0p5x"
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/phaseone:/app/phasetwo:/app/phasethree:/app/phasefour:/app/phasefive"

# Ensure results directories exist
RUN mkdir -p /app/phasefour/results /app/phasefive/results

# Run tests to ensure rigor before running experiments
RUN python /app/phasefour/test_rigor.py

# Default to running the latest phase (Phase Five)
CMD ["python", "phasefive/run_experiments.py"]
