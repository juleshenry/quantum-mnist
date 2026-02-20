# Use a more specific, stable base
FROM tensorflow/tensorflow:2.11.0-gpu

# Combine everything into one single RUN command to minimize layer commits
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    protobuf==3.20.3 \
    tensorflow-quantum \
    cirq \
    sympy \
    pillow \
    numpy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

CMD ["python"]