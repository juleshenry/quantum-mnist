# Phase Three: Generalized N-class Quantum Classifier

This phase extends the binary quantum classifier from Phase One to support multi-class classification.

## Overview

The binary quantum classifier successfully distinguished between two MNIST digits (3 vs 6). This implementation generalizes the approach to handle N classes, making it applicable to datasets like MNIST (10 digits) and plankton (35 species).

## Key Features

### Multi-Class Strategy: One-vs-Rest

The implementation uses a one-vs-rest (OvR) approach:
- Each class gets its own quantum circuit
- Each circuit learns to distinguish one class from all others
- Outputs are combined using softmax for multi-class prediction

### Quantum Circuit Architecture

For each class, the quantum circuit:
1. Encodes the input image as quantum states on a grid of qubits
2. Uses a readout qubit to extract classification information
3. Applies parameterized quantum gates (XX and ZZ gates)
4. Measures the readout qubit to get classification output

### Preprocessing Pipeline

The `preprocess_data` function handles:
- Image resizing (default 4x4 for quantum feasibility)
- Normalization to [0, 1] range
- Binarization using a threshold
- Removal of contradicting examples
- Conversion to quantum circuits
- TensorFlow Quantum tensor conversion

## Files

- `quantum_classifier_n_class.py`: Main implementation of N-class quantum classifier

## Usage

### As a Library

Import and use the quantum classifier in your code:

```python
from phasethree.quantum_classifier_n_class import train_quantum_classifier

# Load your data
x_train, y_train = ...  # Your training data
x_test, y_test = ...    # Your test data

# Train the quantum classifier
model, history, results = train_quantum_classifier(
    x_train, y_train, x_test, y_test,
    num_classes=10,      # Number of classes
    image_size=(4, 4),   # Image dimensions
    epochs=3,            # Training epochs
    batch_size=32        # Batch size
)

print(f"Test accuracy: {results[1]:.4f}")
```

### Standalone Execution

Test on MNIST data:

```bash
python3 phasethree/quantum_classifier_n_class.py
```

This will train and evaluate the quantum classifier on MNIST digits 0-4.

## Key Functions

### `filter_classes(x, y, classes)`
Filter dataset to include only specified classes.

### `preprocess_data(x_train, y_train, x_test, y_test, image_size, threshold)`
Complete preprocessing pipeline for quantum classification.

### `create_quantum_model(image_size)`
Create a single quantum circuit for binary classification.

### `build_quantum_model(num_classes, image_size)`
Build a multi-class quantum model using one-vs-rest approach.

### `train_quantum_classifier(...)`
Complete training pipeline for quantum classification.

## Architecture Details

### Input Encoding
- Images are resized to small dimensions (e.g., 4x4 = 16 qubits)
- Each pixel is binarized and encoded as a qubit state
- Pixel value 1 → |1⟩ state (X gate applied)
- Pixel value 0 → |0⟩ state (no gate)

### Quantum Circuit Layers
- **XX gates**: Entangle data qubits with readout qubit
- **ZZ gates**: Apply phase-based transformations
- **Hadamard gates**: Create superposition on readout qubit

### Output Processing
- PQC (Parametrized Quantum Circuit) layer returns expectation value [-1, 1]
- Multiple PQC outputs are concatenated (one per class)
- Softmax activation produces probability distribution over classes

## Limitations and Considerations

1. **Computational Cost**: Quantum simulation is expensive. Limited to small image sizes (e.g., 4x4).
2. **Scalability**: Number of qubits = image pixels. Larger images require more qubits.
3. **Training Time**: Quantum circuit simulation can be slow, especially for many classes.
4. **Classical Simulation**: Running on classical computers; actual quantum hardware may behave differently.

## Theoretical Basis

The quantum classifier leverages:
- **Quantum superposition**: Exploring multiple states simultaneously
- **Quantum entanglement**: Creating correlations between qubits
- **Quantum interference**: Amplifying correct classifications
- **Variational quantum algorithms**: Optimizing circuit parameters

## Comparison to Binary Classifier

| Aspect | Binary (Phase 1) | Multi-class (Phase 3) |
|--------|------------------|----------------------|
| Classes | 2 | N (any number) |
| Circuits | 1 | N (one per class) |
| Output | Single value | Softmax over N values |
| Loss | Hinge loss | Categorical cross-entropy |

## Dependencies

See `requirements.txt` in the project root.

## References

- [TensorFlow Quantum Tutorial](https://www.tensorflow.org/quantum/tutorials/mnist)
- [Variational Quantum Classifiers](https://arxiv.org/abs/1804.00633)
- Original implementation based on Phase One binary classifier
