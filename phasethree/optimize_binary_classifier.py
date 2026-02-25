
import os
import numpy as np
import tensorflow as tf
import cirq
import sympy
import itertools
from phasetwo.plankton_ingress import prepare_binary_dataset, get_plankton_names
import tensorflow_quantum as tfq

# Configuration
QUBIT_DIMS = (4, 4)

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def convert_to_circuit(image, encoding='angle'):
    # Downsample from 16x16 to 4x4
    image_4x4 = tf.image.resize(image[..., np.newaxis], (4, 4)).numpy().squeeze()
    values = np.ndarray.flatten(image_4x4)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    
    if encoding == 'basis':
        for i, value in enumerate(values):
            if value > 0.5:
                circuit.append(cirq.X(qubits[i]))
    elif encoding == 'angle':
        for i, value in enumerate(values):
            circuit.append(cirq.ry(np.pi * value)(qubits[i]))
            
    return circuit

def create_quantum_model(n_layers=1):
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()
    
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    
    for l in range(n_layers):
        builder.add_layer(circuit, cirq.XX, f"xx{l}")
        builder.add_layer(circuit, cirq.ZZ, f"zz{l}")
    
    circuit.append(cirq.H(readout))
    return circuit, cirq.Z(readout)

def hinge_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true > 0.0, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

# Define the hyperparameter search space
hyperparams = {
    'encoding': ['basis', 'angle'],
    'n_layers': [1, 2],
    'learning_rate': [0.01, 0.001],
    'batch_size': [16, 32]
}

def setup_sweep():
    keys, values = zip(*hyperparams.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total combinations to explore: {len(combinations)}")
    return combinations

def run_sweep(combinations, class_a, class_b):
    best_accuracy = 0
    best_config = None
    
    (x_train, y_train), (x_test, y_test) = prepare_binary_dataset(class_a, class_b, limit=100)
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0
    
    for i, config in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: {config}")
        
        x_train_tfq = tfq.convert_to_tensor([convert_to_circuit(x, encoding=config['encoding']) for x in x_train])
        x_test_tfq = tfq.convert_to_tensor([convert_to_circuit(x, encoding=config['encoding']) for x in x_test])
        
        model_circuit, model_readout = create_quantum_model(n_layers=config['n_layers'])
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(model_circuit, model_readout),
        ])
        
        model.compile(
            loss=tf.keras.losses.Hinge(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            metrics=[hinge_accuracy]
        )
        
        history = model.fit(
            x_train_tfq, y_train_hinge,
            batch_size=config['batch_size'],
            epochs=5,
            verbose=0,
            validation_data=(x_test_tfq, y_test_hinge)
        )
        
        val_acc = max(history.history['val_hinge_accuracy'])
        print(f"Best Val Acc: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = config
            
    return best_config, best_accuracy

if __name__ == "__main__":
    plank = get_plankton_names()
    if len(plank) < 2:
        print("Not enough plankton classes found.")
    else:
        class_a, class_b = plank[0], plank[3] # aphanizomenon vs bosmina
        print(f"Optimizing Quantum Model for {class_a} vs {class_b}")
        
        combos = setup_sweep()
        best_cfg, best_acc = run_sweep(combos, class_a, class_b)
        
        print("\n--- QUANTUM SWEEP COMPLETE ---")
        print(f"Best Configuration: {best_cfg}")
        print(f"Best Accuracy: {best_acc:.4f}")
