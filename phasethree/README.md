# Phase Three: Generalized Multi-Class Quantum Classifier

## Overview

This phase implements a generalization of the binary quantum classifier from Phase One to handle multi-class classification for all MNIST digits (0-9).

## Key Generalization: Binary to Multi-Class

### Binary Classifier (Phase One)
- **Input**: Images of digits 3 and 6
- **Output**: Single readout qubit measuring Z expectation value
- **Loss**: Hinge loss (binary classification)
- **Architecture**: 16 data qubits (4×4 grid) + 1 readout qubit

### Multi-Class Classifier (Phase Three)
- **Input**: Images of all digits (0-9)
- **Output**: 10 readout qubits (one per class)
- **Loss**: Sparse categorical cross-entropy
- **Architecture**: 16 data qubits (4×4 grid) + 10 readout qubits

## Quantum AI Generalization Approach

The generalization from binary (yes/no) to multi-class classification in quantum AI uses the following approach:

1. **Multiple Readout Qubits**: Instead of a single readout qubit for binary classification, we use one readout qubit per class (10 qubits for digits 0-9).

2. **Expectation Values as Logits**: Each readout qubit's expectation value serves as a logit for its corresponding class. The Z-measurement expectation value ranges from -1 to +1.

3. **Parameterized Quantum Circuit (PQC)**: The circuit includes parameterized two-qubit gates (XX and ZZ gates) that create entanglement between data qubits and readout qubits. These parameters are optimized during training.

4. **Multi-Class Loss**: We use sparse categorical cross-entropy loss with logits, treating the expectation values from each readout qubit as the model's output logits.

## Architecture Details

### Data Encoding
- Binary amplitude encoding: Each pixel value is binarized (threshold = 0.5)
- Qubit states: |1⟩ for bright pixels (value > threshold), |0⟩ for dark pixels
- 4×4 images → 16 qubits in a grid layout

### Quantum Circuit Structure
```
1. Initialize readout qubits: X followed by H gates (creates |−⟩ state)
2. Parameterized XX layer: Entangle data qubits with each readout qubit
3. Parameterized ZZ layer: Additional entanglement layer
4. Final H gates on readout qubits
5. Measure Z expectation value on each readout qubit
```

### Circuit Parameters
- **Data qubits**: 16 (4×4 grid)
- **Readout qubits**: 10 (one per class)
- **Parameters per layer**: 16 data × 10 readout = 160 parameters
- **Total parameters**: 2 layers × 160 = 320 trainable parameters

## Usage

Run the multi-class quantum classifier:

```bash
python3 phasethree/quantum_classifier_n_class.py
```

### Requirements
- TensorFlow 2.7.0
- TensorFlow Quantum 0.7.2
- Cirq
- NumPy

## Implementation Highlights

### CircuitLayerBuilder Class
Builds parameterized quantum circuit layers with entangling gates between data and readout qubits:
```python
class CircuitLayerBuilder:
    def add_layer(self, circuit, gate, prefix):
        for j, readout in enumerate(self.readout_qubits):
            for i, data_qubit in enumerate(self.data_qubits):
                symbol = sympy.Symbol(f'{prefix}_r{j}_d{i}')
                circuit.append(gate(data_qubit, readout)**symbol)
```

### Multi-Readout Architecture
```python
readout_qubits = [cirq.GridQubit(-1, i) for i in range(num_classes)]
readout_operators = [cirq.Z(readout) for readout in readout_qubits]
```

### Model Compilation
```python
quantum_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
    metrics=['accuracy']
)
```

## Comparison with Classical Baseline

The implementation includes a "fair" classical neural network with similar parameter count for comparison:
- Input: Flattened 4×4 image (16 features)
- Hidden layer: 2 neurons with ReLU activation
- Output layer: 10 neurons (one per class)

## Expected Results

The quantum classifier demonstrates multi-class classification capability on MNIST digits 0-9, showing that:
1. Binary quantum classifiers can be generalized to multi-class problems
2. Multiple readout qubits enable natural multi-class classification in quantum circuits
3. The approach scales to more classes by adding more readout qubits

## Quantum AI Terminology

- **PQC (Parameterized Quantum Circuit)**: A quantum circuit with trainable parameters that can be optimized
- **Readout Qubit**: A qubit whose measurement provides the model output
- **Expectation Value**: The average measurement outcome for a quantum operator (e.g., Z)
- **Amplitude Encoding**: Encoding classical data as quantum amplitudes
- **Binary Encoding**: Encoding classical data as computational basis states (|0⟩ or |1⟩)

## Next Steps (Phase Four)

Apply this generalized multi-class quantum classifier to the plankton dataset for real-world classification tasks.
