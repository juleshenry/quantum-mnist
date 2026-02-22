# 1. Use a Python 3.10 base instead of Ubuntu 24.04 (Python 3.12)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# 2. Install essential build tools (TFQ needs these for some C++ components)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Upgrade core pip tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. Install TensorFlow & TFQ in one go to let the resolver find a match
# Note: TFQ 0.7.3+ usually pairs well with TF 2.15.0 or 2.16.1
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    tensorflow-quantum

# 5. Legacy Keras environment variable (Still needed for many TFQ operations)
ENV TF_USE_LEGACY_KERAS=1

# 6. Remaining dependencies
RUN pip install --no-cache-dir pillow scikit-learn matplotlib

WORKDIR /app
COPY quantum_binary_classifier.py /app/

CMD ["python", "quantum_binary_classifier.py", "--help"]