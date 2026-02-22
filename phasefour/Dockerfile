# Keep your base image
FROM tensorflow/tensorflow:2.11.0

# Install build-essential for C++ compatibility
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    # TFQ 0.7.2 is the correct match for TF 2.11.0
    # We also pin Cirq to 0.14.1 which is the stable partner for this TFQ version
    pip install --no-cache-dir \
    tensorflow-quantum==0.7.2 \
    cirq==0.14.1 \
    sympy \
    pillow \
    numpy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

CMD ["python"]