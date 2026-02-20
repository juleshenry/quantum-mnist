# Use a base image that matches the TensorFlow version required by TFQ
FROM tensorflow/tensorflow:2.11.0-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Fix the Protobuf issue by pinning a compatible version (3.20.x is usually the sweet spot for TF 2.11)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    protobuf==3.20.3 \
    tensorflow==2.11.0 \
    tensorflow-quantum \
    cirq \
    sympy \
    pillow \
    numpy

CMD ["python"]