# Use the older but most compatible version for TFQ
FROM tensorflow/tensorflow:2.7.0

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow==2.7.0 \
    tensorflow-quantum==0.6.1 \
    cirq==0.13.1 \
    sympy==1.8 \
    pillow \
    numpy==1.21.6 \
    matplotlib \
    seaborn \
    scikit-learn==1.0.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project
COPY . .

# Set python path to include src
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "scripts/run_unified_classifier.py"]