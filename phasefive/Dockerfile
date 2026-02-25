FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    pandas==1.3.5 \
    scikit-learn==1.0.2 \
    matplotlib==3.5.1 \
    pillow==9.0.1 \
    seaborn==0.11.2 \
    tensorflow==2.7.0 \
    tensorflow-quantum==0.7.2 \
    cirq==0.13.1 \
    sympy==1.8

# Copy data and phasefive code
COPY data /app/data
COPY phasefive /app/phasefive

WORKDIR /app/phasefive

# Set environment variables
ENV DATA_DIR="/app/data/zooplankton_0p5x"
ENV PYTHONPATH="/app/phasefive"

# Ensure results directory exists
RUN mkdir -p /app/phasefive/results

CMD ["python", "run_experiments.py"]
