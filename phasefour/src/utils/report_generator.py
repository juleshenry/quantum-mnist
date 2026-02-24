"""
Comparison Report: Classical vs Quantum Plankton Classification

This script generates a comparison between the classical approach (Phase Two)
and the quantum approach (Phase Four) for plankton image classification.
"""

def generate_comparison_report():
    """
    Generate a markdown report comparing classical and quantum approaches.
    """
    
    report = """
# Classical vs Quantum Plankton Classification Comparison

## Overview

This report compares two approaches for binary plankton classification:
- **Phase Two**: Classical feedforward neural network (FFN)
- **Phase Four**: Hybrid quantum-classical neural network

---

## Architecture Comparison

### Phase Two: Classical FFN

**Architecture:**
```
Input: 16×16 grayscale images (256 pixels)
  ↓
Flatten: 256 features
  ↓
Dense Layer: 2 neurons (ReLU activation)
  ↓
Output Layer: 1 neuron (sigmoid/linear)
```

**Parameters:**
- Input size: 256 features (16×16 pixels)
- Hidden neurons: 2
- Total trainable parameters: ~514
  - Layer 1: 256 × 2 + 2 = 514 parameters
  - Layer 2: 2 × 1 + 1 = 3 parameters

**Training:**
- Batch size: 32
- Optimizer: Adam
- Loss: Binary cross-entropy
- Epochs: 20

**Advantages:**
- Fast training and inference
- Simple architecture
- Well-understood training dynamics

**Limitations:**
- Limited expressivity (only 2 hidden neurons)
- Linear decision boundary
- Struggles with complex patterns

---

### Phase Four: Quantum-Classical Hybrid

**Architecture:**
```
Input: 8×8 grayscale images (64 pixels)
  ↓
Binary Threshold Encoding: 64 qubits
  ↓
Quantum Circuit (PQC):
  - Readout qubit preparation (|+⟩ state)
  - XX entangling layer (64 parameters)
  - ZZ entangling layer (64 parameters)
  ↓
Measurement: Z operator on readout qubit
  ↓
Classical Dense: 1 neuron
```

**Parameters:**
- Input size: 64 qubits (8×8 pixels)
- Quantum parameters: 128 (2 layers × 64 qubits)
- Classical parameters: 2 (1 weight + 1 bias)
- Total trainable parameters: ~130

**Quantum Circuit Details:**
- Data qubits: 64 (8×8 grid)
- Readout qubit: 1 ancilla qubit
- Entangling gates: XX and ZZ between each data qubit and readout
- Circuit depth: ~4 (including preparation and measurement)

**Training:**
- Batch size: 4-8 (smaller due to quantum simulation overhead)
- Optimizer: Adam (learning rate: 0.02)
- Loss: Binary cross-entropy
- Epochs: 10-15

**Advantages:**
- Parameter-efficient encoding
- Captures quantum correlations via entanglement
- Potentially exponential expressivity with qubit count
- Novel approach exploring quantum advantage

**Limitations:**
- Lower resolution (8×8 vs 16×16)
- Slower training due to quantum simulation
- Requires specialized hardware/software
- Limited by current quantum simulators

---

## Detailed Comparison Table

| Aspect | Classical (Phase Two) | Quantum (Phase Four) |
|--------|----------------------|---------------------|
| **Image Resolution** | 16×16 (256 pixels) | 8×8 (64 pixels) |
| **Input Features** | 256 real-valued | 64 binary-encoded |
| **Model Type** | Feedforward NN | Hybrid Quantum-Classical |
| **Trainable Parameters** | ~517 | ~130 |
| **Hidden Representation** | 2 neurons | 64 qubits + entanglement |
| **Batch Size** | 32 | 4-8 |
| **Training Time/Epoch** | Fast (~seconds) | Moderate (~minutes) |
| **Inference Time** | Very Fast | Moderate |
| **Hardware Requirements** | CPU/GPU | Quantum simulator or QPU |
| **Scalability** | Easy (add layers/neurons) | Limited by qubit count |
| **Interpretability** | Moderate | Low (quantum black box) |
| **Research Novelty** | Standard approach | Cutting-edge research |

---

## Encoding Strategy Comparison

### Classical Encoding
- **Method**: Direct pixel value usage
- **Range**: [0, 1] continuous values
- **Information**: Full grayscale spectrum
- **Pros**: No information loss, straightforward
- **Cons**: High-dimensional input space

### Quantum Encoding
- **Method**: Binary threshold encoding
- **Range**: {0, 1} binary states → |0⟩ or |1⟩
- **Information**: Binary classification of pixels (dark/bright)
- **Pros**: Natural mapping to qubit states, efficient for quantum
- **Cons**: Information loss from binarization

---

## Theoretical Considerations

### Classical Model Capacity
With 2 hidden neurons, the classical model can approximate:
- 2 linear decision boundaries
- Limited non-linear combinations

**Expressivity**: O(hidden_neurons)

### Quantum Model Capacity
With 64 qubits and entangling layers:
- Hilbert space dimension: 2^64 ≈ 10^19
- Parameterized gates create complex superpositions
- Entanglement enables non-local correlations

**Expressivity**: Potentially O(2^qubits) for certain problems

---

## Expected Performance

### Phase Two (Classical):
Based on "fair" FFN experiments:
- **Accuracy**: 65-75% (varies by species pair)
- **Training**: Converges in 10-20 epochs
- **Reliability**: Consistent results

### Phase Four (Quantum):
Based on quantum image classification literature:
- **Accuracy**: 70-85% (predicted, varies by species pair)
- **Training**: Converges in 10-15 epochs
- **Reliability**: May have higher variance due to quantum effects

**Hypothesis**: Quantum approach may perform better on:
- Species with distinctive patterns (high contrast)
- Binary features (presence/absence of structures)
- Cases where classical model underfits

---

## Implementation Complexity

### Classical Implementation (Phase Two):
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(16,16,1)))
model.add(tf.keras.layers.Dense(2, activation='relu'))
model.add(tf.keras.layers.Dense(1))
```
**Lines of code**: ~20
**Dependencies**: TensorFlow

### Quantum Implementation (Phase Four):
```python
# Quantum circuit creation
data_qubits = cirq.GridQubit.rect(8, 8)
readout = cirq.GridQubit(-1, -1)
circuit = cirq.Circuit()
# ... parameterized layers ...

# Hybrid model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(model_circuit, model_readout),
    tf.keras.layers.Dense(1)
])
```
**Lines of code**: ~500+ (with helpers)
**Dependencies**: TensorFlow, TensorFlow Quantum, Cirq, Sympy

---

## Practical Considerations

### When to Use Classical (Phase Two):
- ✓ Need fast training/inference
- ✓ Limited computational resources
- ✓ Production deployment required
- ✓ Standard accuracy sufficient
- ✓ Interpretability important

### When to Use Quantum (Phase Four):
- ✓ Research/experimental context
- ✓ Exploring quantum advantage
- ✓ Have access to quantum hardware/simulators
- ✓ Parameter efficiency critical
- ✓ Novel approach desired
- ✓ Pattern recognition via quantum correlations

---

## Future Directions

### Classical Path:
1. Increase hidden layer size (2 → 10+ neurons)
2. Add more layers (deep learning)
3. Use convolutional layers for spatial features
4. Ensemble methods

### Quantum Path:
1. Increase image resolution (8×8 → 12×12)
2. Add more entangling layers
3. Try different encoding strategies (amplitude, angle)
4. Explore quantum kernel methods
5. Deploy on real quantum hardware
6. Multi-class quantum classification

---

## Conclusion

Both approaches demonstrate viability for plankton classification:

**Classical FFN** provides a simple, interpretable baseline with fast training but limited expressivity with only 2 hidden neurons.

**Quantum Hybrid** offers a novel, parameter-efficient approach that explores quantum advantage through entanglement, though at the cost of increased complexity and simulation overhead.

The quantum approach is particularly interesting for:
- Research into quantum machine learning
- Scenarios where parameter efficiency matters
- Exploration of quantum correlations in image data

For production use, classical approaches remain more practical, but the quantum implementation demonstrates the feasibility and potential of quantum image classification on real-world biological datasets.

---

## References

1. Phase One implementation: `phaseone/quantum_image_mnist.ipynb`
2. Phase Two implementation: `phasetwo/PhaseTwo.ipynb`
3. Phase Four implementation: `src/classifiers/plankton_classifier.py`
4. Technical documentation: `README.md`
"""
    
    return report


def save_comparison_report(output_path="COMPARISON_REPORT.md"):
    """Save comparison report to file."""
    report = generate_comparison_report()
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to: {output_path}")
    print(f"Report length: {len(report)} characters")


if __name__ == "__main__":
    import os
    
    # Save to phasefour directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "COMPARISON_REPORT.md")
    
    save_comparison_report(output_path)
