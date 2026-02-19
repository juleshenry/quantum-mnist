# Use Python 3.9 as requested
FROM python:3.9-slim

# Install system dependencies (required for some quantum/imaging libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first to avoid the notice in your logs
RUN pip install --no-cache-dir --upgrade pip

# TFQ is very sensitive to the TF version. 
# As of current releases, TFQ 0.7.3 pairs with TF 2.11.0.
RUN pip install --no-cache-dir \
    tensorflow==2.11.0 \
    tensorflow-quantum \
    cirq \
    sympy \
    pillow \
    numpy

CMD ["python"]