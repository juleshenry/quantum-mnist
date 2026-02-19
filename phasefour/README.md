# Phase Four: Quantum Classification for Plankton Dataset

This phase applies quantum image classification techniques to the plankton dataset, extending the quantum MNIST approach to handle more complex, real-world biological images.

## Overview

The quantum plankton classifier implements a hybrid quantum-classical neural network using:
- **Cirq** for quantum circuit construction
- **TensorFlow Quantum** for hybrid model training
- **Parameterized Quantum Circuits (PQC)** for learning

## Architecture

### Quantum Encoding Strategy

**Binary Threshold Encoding:**
- Images are preprocessed to grayscale and resized to quantum-compatible dimensions
- Each pixel maps to one qubit in a grid layout
- Pixel values > 0.5 → |1⟩ state (X gate applied)
- Pixel values ≤ 0.5 → |0⟩ state (no operation)

### Image Size Considerations

| Size | Qubits | Simulation | Recommendation |
|------|--------|------------|----------------|
| 4×4  | 16     | Very Fast  | Too small for plankton features |
| 8×8  | 64     | Fast       | **Recommended** for development/testing |
| 16×16| 256    | Very Slow  | May be infeasible for simulation |

The 8×8 configuration provides a good balance between quantum circuit complexity and model expressiveness for plankton classification.

### Quantum Circuit Architecture

```
1. Data Encoding Layer
   - Grid of data qubits (e.g., 8×8 = 64 qubits)
   - Binary encoding of image pixels
   
2. Readout Qubit Preparation
   - Single ancilla qubit initialized to |+⟩ state
   - Acts as measurement interface
   
3. Parameterized Entangling Layers
   - XX gates: Create entanglement between data and readout qubits
   - ZZ gates: Add phase relationships
   - Each gate parameterized by trainable symbols
   
4. Readout Measurement
   - Measure expectation value of Z operator on readout qubit
   - Value in range [-1, 1] fed to classical layer
   
5. Classical Post-Processing
   - Dense layer maps quantum output to binary classification
```

## Implementation Details

### Key Components

**PlanktonQuantumClassifier:**
- Main class orchestrating quantum image classification
- Handles image preprocessing, circuit encoding, and model training
- Configurable image dimensions for quantum encoding

**CircuitLayerBuilder:**
- Helper class for constructing parameterized quantum layers
- Applies 2-qubit gates between each data qubit and the readout qubit
- Supports XX, ZZ, and other parameterized gate types

### Image Preprocessing Pipeline

```python
1. Load raw plankton image (variable size, RGB)
2. Convert to grayscale (L mode in PIL)
3. Resize using bilinear interpolation to target dimensions
4. Normalize pixel values to [0, 1] range
5. Apply binary threshold for quantum encoding
```

### Training Configuration

**Hyperparameters:**
- Learning rate: 0.02 (Adam optimizer)
- Batch size: 4-16 (smaller due to quantum circuit overhead)
- Epochs: 10-20
- Train/test split: 75/25

**Loss Function:**
- Binary cross-entropy with logits
- Suitable for binary classification tasks

## Usage

### Basic Example: Single Pair Classification

```python
from plankton_quantum_algorithm import PlanktonQuantumClassifier

# Initialize classifier with 8x8 images (64 qubits)
classifier = PlanktonQuantumClassifier(image_size=(8, 8))

# Train on two plankton categories
model, history, accuracy = classifier.train_binary_classifier(
    category_a="bosmina",
    category_b="cyclops",
    plankton_dir="data/zooplankton_0p5x",
    max_images=30,
    epochs=15,
    batch_size=8
)

print(f"Test Accuracy: {accuracy:.4f}")
```

### Cartesian Product Comparison

```python
from plankton_quantum_algorithm import run_cartesian_comparison

# Run on multiple category pairs
results = run_cartesian_comparison(
    plankton_dir='data/zooplankton_0p5x',
    image_size=(8, 8),
    max_pairs=10,
    max_images=30,
    epochs=15
)
```

This approach mirrors the "fair" comparison strategy used in Phase Two's classical neural network, but with quantum circuits.

## Installation Requirements

```bash
pip install tensorflow tensorflow-quantum cirq sympy pillow numpy
```

**Note:** TensorFlow Quantum requires:
- TensorFlow 2.x
- Python 3.7-3.9
- Compatible with specific TensorFlow versions (check TFQ compatibility)

## Comparison to Classical Approach

### Phase Two (Classical FFN):
- Input: Flattened 16×16 = 256 features
- Hidden layer: 2 neurons with ReLU
- Output: 1 neuron (binary classification)
- Total parameters: ~514

### Phase Four (Quantum):
- Input: Quantum circuit with 64 qubits (8×8)
- Quantum layer: 128 trainable parameters (2 layers × 64 qubits)
- Classical output: 1 neuron
- Total parameters: ~129

**Advantages of Quantum Approach:**
- Potentially captures complex correlations through entanglement
- Parameter-efficient encoding of high-dimensional data
- Explores quantum advantage for pattern recognition

**Challenges:**
- Simulation overhead for large quantum circuits
- Training can be slower than classical models
- Limited to binary classification in current implementation

## Results and Observations

### Expected Performance

Based on quantum image classification literature:
- Binary classification accuracy: 70-85% (depending on species similarity)
- Training convergence: 10-20 epochs typically sufficient
- Model size: Smaller parameter count than classical approaches

### Known Limitations

1. **Simulation Constraints:** 
   - 8×8 (64 qubits) is practical limit for CPU simulation
   - 16×16 (256 qubits) requires significant computational resources

2. **Image Resolution:**
   - Lower resolution than classical models (8×8 vs 16×16+)
   - May miss fine-grained plankton features

3. **Binary Classification:**
   - Current implementation limited to pairwise comparisons
   - Multi-class quantum classification requires extension

## Future Directions

### Short-term Improvements

1. **Gradient-free optimization:**
   - Try parameter-shift rule for gradient computation
   - Experiment with quantum natural gradient

2. **Circuit depth optimization:**
   - Add more entangling layers for 8×8 images
   - Test different gate sequences (XY, CZ, etc.)

3. **Hybrid architectures:**
   - Pre-processing with classical CNN
   - Quantum circuit on extracted features

### Long-term Research

1. **Multi-class quantum classification:**
   - Extend to N-way classification
   - Implement quantum decision trees or hierarchical models

2. **Quantum advantage demonstration:**
   - Compare against matched-parameter classical models
   - Identify tasks where quantum provides speedup

3. **Real quantum hardware:**
   - Deploy on NISQ devices (IBM, Google, Rigetti)
   - Handle noise and error mitigation

## References

1. **Quantum MNIST:** Original paper demonstrating quantum image classification on handwritten digits
   - https://thesai.org/Downloads/Volume11No10/Paper_40-Handwritten_Numeric_Image_Classification.pdf

2. **Plankton Classification:** Classical deep learning approaches
   - https://arxiv.org/pdf/2108.05258.pdf

3. **Quantum Machine Learning:** Theoretical foundations
   - https://arxiv.org/pdf/2011.02831.pdf

4. **TensorFlow Quantum:** Framework documentation
   - https://www.tensorflow.org/quantum

## Contact and Contributions

This implementation is part of the quantum-mnist project exploring quantum machine learning for image classification. Contributions and improvements are welcome!

```
docker build --platform linux/amd64 -t my-quantum-app .

docker run -it quantum-env

docker run -it quantum-env python example_usage.py
```