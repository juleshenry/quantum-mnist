# Dockerfile
FROM tensorflow/tensorflow:2.12.0-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip
RUN pip install tensorflow-quantum
RUN pip install cirq
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install Pillow
RUN pip install pandas
RUN pip install seaborn

# Create working directory
WORKDIR /workspace

# Copy the quantum classifier script
COPY quantum_image_classifier.py .

# Default command
CMD ["python", "quantum_image_classifier.py"]