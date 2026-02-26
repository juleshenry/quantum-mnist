import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

def create_k_category_quantum_model(k, n_layers=1):
    # Total 17 qubits: 16 data + 1 readout
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    
    circuit = cirq.Circuit()
    
    # Entanglement layer
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i+1]))
    circuit.append(cirq.CZ(data_qubits[-1], readout))
    
    # Parametric layers
    for l in range(n_layers):
        for i, q in enumerate(data_qubits):
            # XX gates
            symbol_xx = sympy.Symbol(f'xx-{l}-{i}')
            circuit.append(cirq.XX(q, readout)**symbol_xx)
            # ZZ gates
            symbol_zz = sympy.Symbol(f'zz-{l}-{i}')
            circuit.append(cirq.ZZ(q, readout)**symbol_zz)

    # Observables: We need k observables for k categories
    # We use the readout qubit (index 0) and data qubits (indices 1 to k-1)
    all_qubits = [readout] + data_qubits
    observables = [cirq.Z(all_qubits[i]) for i in range(k)]
        
    return circuit, observables

def convert_to_circuit(image):
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit

class MultiClassPQC(tf.keras.layers.Layer):
    def __init__(self, circuit, observables, **kwargs):
        super().__init__(**kwargs)
        self.pqc = tfq.layers.PQC(circuit, observables)
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self, inputs):
        expectations = self.pqc(inputs)
        return self.softmax(expectations)

def create_qnn_multiclass_model(k, n_layers=1, learning_rate=0.01):
    circuit, observables = create_k_category_quantum_model(k, n_layers)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        MultiClassPQC(circuit, observables)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model
