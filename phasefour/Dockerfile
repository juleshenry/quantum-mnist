FROM python:3.9-slim

WORKDIR /app/phasefour

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    pillow \
    tensorflow==2.7.0 \
    tensorflow-quantum==0.7.2 \
    cirq==0.13.1 \
    sympy==1.8

# Copy the source code and data
COPY phasefour /app/phasefour
COPY data /app/data

# Ensure output directory exists
RUN mkdir -p /app/phasefour/results

# Set python path to find data_loader
ENV PYTHONPATH="/app/phasefour"

CMD ["python", "run_experiments.py"]
