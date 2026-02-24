import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import pandas as pd
import os
from phasetwo.plankton_ingress import prepare_binary_dataset

def angle_encode(image_4x4):
    """Encodes a 4x4 image into a quantum circuit using angle encoding."""
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, val in enumerate(image_4x4.flatten()):
        # Map [0, 1] to [0, pi]
        circuit.append(cirq.ry(np.pi * val)(qubits[i]))
    return circuit

def create_quantum_model():
    """Creates a quantum model circuit and readout operator."""
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()
    
    # Initialize readout qubit
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    # Parameters for the model
    # Layer 1: Data-Readout interaction
    for i, qubit in enumerate(data_qubits):
        symbol_xx = sympy.Symbol(f'xx1_{i}')
        symbol_zz = sympy.Symbol(f'zz1_{i}')
        circuit.append(cirq.XX(qubit, readout)**symbol_xx)
        circuit.append(cirq.ZZ(qubit, readout)**symbol_zz)
        
    # Layer 2: Entanglement between data qubits to spread information
    for i in range(len(data_qubits)):
        circuit.append(cirq.CNOT(data_qubits[i], data_qubits[(i+1)%len(data_qubits)]))

    # Layer 3: More parameterized interactions
    for i, qubit in enumerate(data_qubits):
        symbol_ry = sympy.Symbol(f'ry2_{i}')
        circuit.append(cirq.ry(symbol_ry)(qubit))
        
    # Final readout step
    circuit.append(cirq.H(readout))
    
    return circuit, cirq.Z(readout)

def hinge_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true > 0.0, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

def run_improved_classification(class_a, class_b):
    print(f"\n--- Training Improved Model for {class_a} vs {class_b} ---")
    
    # Load more data: 150 samples per class
    (x_train, y_train), (x_test, y_test) = prepare_binary_dataset(class_a, class_b, limit=150)
    
    # Downsample to 4x4
    x_train_4x4 = tf.image.resize(x_train[..., np.newaxis], (4, 4)).numpy().squeeze()
    x_test_4x4 = tf.image.resize(x_test[..., np.newaxis], (4, 4)).numpy().squeeze()
    
    # Convert to circuits
    x_train_circ = [angle_encode(x) for x in x_train_4x4]
    x_test_circ = [angle_encode(x) for x in x_test_4x4]
    
    # Convert to tensors
    x_train_tfq = tfq.convert_to_tensor(x_train_circ)
    x_test_tfq = tfq.convert_to_tensor(x_test_circ)
    
    # Labels to hinge format
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0
    
    model_circuit, model_readout = create_quantum_model()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])
    
    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
        metrics=[hinge_accuracy]
    )
    
    # Increase epochs to 20
    model.fit(
        x_train_tfq, y_train_hinge,
        batch_size=32,
        epochs=20,
        verbose=0,
        validation_data=(x_test_tfq, y_test_hinge)
    )
    
    results = model.evaluate(x_test_tfq, y_test_hinge, verbose=0)
    print(f"Results for {class_a} vs {class_b}: Accuracy = {results[1]:.4f}")
    return results[1]

if __name__ == "__main__":
    pairs = [
        ('aphanizomenon', 'bosmina'),
        ('dinobryon', 'nauplius'),
        ('maybe_cyano', 'diaphanosoma'),
        ('asterionella', 'uroglena'),
        ('cyclops', 'ceratium'),
        ('daphnia', 'keratella_cochlearis')
    ]
    
    summary = []
    for a, b in pairs:
        try:
            acc = run_improved_classification(a, b)
            summary.append({'pair': f"{a}_vs_{b}", 'accuracy': acc})
        except Exception as e:
            print(f"Error for {a} vs {b}: {e}")
            
    df = pd.DataFrame(summary)
    print("
--- FINAL SUMMARY ---")
    print(df)
    
    success_count = len(df[df['accuracy'] >= 0.60])
    print(f"
Pairs with accuracy >= 60%: {success_count}")
    
    if success_count >= 5:
        print("GOAL ACHIEVED!")
    else:
        print("Goal not achieved yet. More improvements needed.")

    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/improved_results.csv', index=False)
