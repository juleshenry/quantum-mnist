# Use the older but most compatible version for TFQ
FROM tensorflow/tensorflow:2.7.0

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    # We must pin these exactly to avoid the 'undefined symbol' error
    pip install --no-cache-dir \
    tensorflow==2.7.0 \
    tensorflow-quantum==0.6.1 \
    cirq==0.13.1 \
    sympy==1.8 \
    pillow \
    numpy==1.21.6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

CMD ["python"]