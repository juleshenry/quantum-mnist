# Quantum Classifier Architecture: Binary vs Multi-Class

## Binary Classification (Phase One)

```
Input: MNIST images (3 or 6)
   ↓
4×4 Binary Encoding (16 qubits)
   ↓
[q0] [q1] [q2] [q3]     ← Data Qubits (4×4 grid)
[q4] [q5] [q6] [q7]
[q8] [q9] [q10] [q11]
[q12] [q13] [q14] [q15]
   ↓  ↓   ↓    ↓
   └──┴───┴────┴──→ [readout] ← Single Readout Qubit
                         ↓
                    Z measurement
                         ↓
                  Expectation value ∈ [-1, +1]
                         ↓
                    Hinge Loss
                         ↓
                  Binary output: 3 or 6?
```

**Parameters**: 2 layers × 16 data qubits × 1 readout = **32 parameters**

## Multi-Class Classification (Phase Three)

```
Input: MNIST images (0-9)
   ↓
4×4 Binary Encoding (16 qubits)
   ↓
[q0] [q1] [q2] [q3]     ← Data Qubits (4×4 grid)
[q4] [q5] [q6] [q7]
[q8] [q9] [q10] [q11]
[q12] [q13] [q14] [q15]
   ↓  ↓   ↓    ↓
   ├──┼───┼────┼──→ [readout_0] ← Readout for digit 0
   ├──┼───┼────┼──→ [readout_1] ← Readout for digit 1
   ├──┼───┼────┼──→ [readout_2] ← Readout for digit 2
   ├──┼───┼────┼──→ [readout_3] ← Readout for digit 3
   ├──┼───┼────┼──→ [readout_4] ← Readout for digit 4
   ├──┼───┼────┼──→ [readout_5] ← Readout for digit 5
   ├──┼───┼────┼──→ [readout_6] ← Readout for digit 6
   ├──┼───┼────┼──→ [readout_7] ← Readout for digit 7
   ├──┼───┼────┼──→ [readout_8] ← Readout for digit 8
   └──┴───┴────┴──→ [readout_9] ← Readout for digit 9
                         ↓
                  Z measurements (10 values)
                         ↓
             [e0, e1, e2, ..., e9] ∈ [-1, +1]^10
                         ↓
                  Softmax (logits)
                         ↓
            Categorical Cross-Entropy Loss
                         ↓
                  Class prediction: 0-9
```

**Parameters**: 2 layers × 16 data qubits × 10 readouts = **320 parameters**

## Key Generalization

The generalization from binary to multi-class is achieved by:

1. **Scaling Readout Qubits**: 1 readout → N readouts (one per class)
2. **Parallel Classification**: Each readout independently evaluates one class
3. **Vector Output**: Single scalar → N-dimensional vector of logits
4. **Loss Function**: Hinge loss → Categorical cross-entropy

## Quantum Circuit Layers

### Layer Structure
```python
for each readout_qubit in [readout_0, ..., readout_9]:
    for each data_qubit in [q0, ..., q15]:
        Apply XX^(θ) gate between data_qubit and readout_qubit
        Apply ZZ^(φ) gate between data_qubit and readout_qubit
```

Where θ and φ are trainable parameters optimized during training.

## Mathematical Representation

### Binary Classifier Output
```
y_pred = ⟨ψ|Z_readout|ψ⟩
       ∈ [-1, +1]
```

### Multi-Class Classifier Output
```
y_pred = [⟨ψ|Z_r0|ψ⟩, ⟨ψ|Z_r1|ψ⟩, ..., ⟨ψ|Z_r9|ψ⟩]
       ∈ [-1, +1]^10
       
class = argmax(softmax(y_pred))
```

## Advantages of Multi-Readout Architecture

1. **Natural Extension**: Direct generalization of binary approach
2. **Parallel Processing**: All classes evaluated simultaneously
3. **Quantum Entanglement**: Data qubits entangled with all readouts
4. **Scalability**: Add more readout qubits for more classes

## Comparison with Classical Approach

### Classical Multi-Class
- Flatten 4×4 image → 16 features
- Hidden layer: 2 neurons
- Output layer: 10 neurons
- Total: ~54 parameters

### Quantum Multi-Class
- Encode 4×4 image → 16 qubits
- Entangle with 10 readout qubits
- 2 parameterized layers
- Total: ~320 parameters

The quantum approach has more parameters but leverages quantum phenomena (superposition, entanglement) for classification.
