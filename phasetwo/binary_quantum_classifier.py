# Note: This script requires tensorflow_quantum (TFQ) and cirq.
# It is designed to be run in a Google Colab environment or a local setup with TFQ.

import cirq
import sympy
import numpy as np
import tensorflow as tf
# import tensorflow_quantum as tfq  # Commented out to prevent local import errors

# Import local data loader
try:
    from phasetwo.plankton_ingress import prepare_binary_dataset, get_plankton_names
except ImportError:
    # Fallback for colab if needed
    pass

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def convert_to_circuit(image):
    """Encode classical image into quantum datapoint using angle encoding."""
    # Downsample from 16x16 to 4x4 for simulation feasibility
    image_4x4 = tf.image.resize(image[..., np.newaxis], (4, 4)).numpy().squeeze()
    
    values = np.ndarray.flatten(image_4x4)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        # Use Angle Encoding (Ry rotation)
        circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit

def create_quantum_model():
    """Create a QNN model circuit with improved expressivity."""
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()
    
    # Add entanglement between data qubits
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i+1]))
    
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    builder.add_layer(circuit, cirq.YY, "yy1") # Added YY layer for more expressivity
    
    circuit.append(cirq.H(readout))
    return circuit, cirq.Z(readout)

def run_quantum_classification(class_a, class_b):
    import tensorflow_quantum as tfq
    
    (x_train, y_train), (x_test, y_test) = prepare_binary_dataset(class_a, class_b, limit=150)
    
    # Convert labels to hinge loss format [-1, 1]
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0
    
    # Convert images to circuits
    x_train_circ = [convert_to_circuit(x) for x in x_train]
    x_test_circ = [convert_to_circuit(x) for x in x_test]
    
    # Convert circuits to tensors
    x_train_tfq = tfq.convert_to_tensor(x_train_circ)
    x_test_tfq = tfq.convert_to_tensor(x_test_circ)
    
    model_circuit, model_readout = create_quantum_model()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])
    
    def hinge_accuracy(y_true, y_pred):
        y_true = tf.cast(y_true > 0.0, tf.float32)
        y_pred = tf.cast(y_pred > 0.0, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[hinge_accuracy]
    )
    
    model.fit(
        x_train_tfq, y_train_hinge,
        batch_size=16,
        epochs=15,
        verbose=1,
        validation_data=(x_test_tfq, y_test_hinge)
    )
    
    return model.evaluate(x_test_tfq, y_test_hinge)

if __name__ == "__main__":
    print("--- Phase 2: Basic Binary Quantum Classification ---")
    plank = get_plankton_names()
    if len(plank) < 2:
        print("Not enough plankton classes.")
    else:
        class_a, class_b = plank[0], plank[3] # aphanizomenon vs bosmina
        print(f"Running Quantum Classification for {class_a} vs {class_b}")
        try:
            results = run_quantum_classification(class_a, class_b)
            print(f"\nQuantum Results - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        except Exception as e:
            print(f"Error during quantum execution: {e}")
            print("Note: This requires a working tensorflow_quantum installation.")
