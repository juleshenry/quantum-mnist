# Use the older but most compatible version for TFQ
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

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow==2.7.0 \
    tensorflow-quantum==0.7.2 \
    cirq==0.13.1 \
    sympy==1.8 \
    pillow \
    numpy==1.21.6 \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn==1.0.2

WORKDIR /app

# Copy the entire project
COPY . .

# Set python path to include the root and all phases
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/phaseone:/app/phasetwo:/app/phasethree:/app/phasefour"

# Ensure results directory exists for phase four
RUN mkdir -p /app/phasefour/results

# Default command
CMD ["python"]
