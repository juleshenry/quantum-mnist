
import os
import numpy as np
import tensorflow as tf
import cirq
import sympy
from phasetwo.plankton_ingress import prepare_binary_dataset, get_plankton_names
import tensorflow_quantum as tfq

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def convert_to_circuit(image):
    # Downsample from 16x16 to 4x4
    image_4x4 = tf.image.resize(image[..., np.newaxis], (4, 4)).numpy().squeeze()
    values = np.ndarray.flatten(image_4x4 > 0.5)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def create_quantum_model():
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    circuit.append(cirq.H(readout))
    return circuit, cirq.Z(readout)

def hinge_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true > 0.0, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

def evaluate_pair(class_a, class_b):
    (x_train, y_train), (x_test, y_test) = prepare_binary_dataset(class_a, class_b, limit=100)
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0
    x_train_tfq = tfq.convert_to_tensor([convert_to_circuit(x) for x in x_train])
    x_test_tfq = tfq.convert_to_tensor([convert_to_circuit(x) for x in x_test])
    
    model_circuit, model_readout = create_quantum_model()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])
    model.compile(loss=tf.keras.losses.Hinge(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[hinge_accuracy])
    
    model.fit(x_train_tfq, y_train_hinge, batch_size=32, epochs=5, verbose=0)
    loss, acc = model.evaluate(x_test_tfq, y_test_hinge, verbose=0)
    return acc

if __name__ == "__main__":
    plank = get_plankton_names()
    pairs = [
        (plank[0], plank[3]), # aphanizomenon vs bosmina
        (plank[1], plank[2]), # asplanchna vs asterionella
        (plank[4], plank[5]), # brachionus vs ceratium
        (plank[6], plank[7]), # chaoborus vs conochilus
        (plank[8], plank[9]), # copepod_skins vs cyclops
    ]
    
    for ca, cb in pairs:
        acc = evaluate_pair(ca, cb)
        print(f"Baseline Accuracy for {ca} vs {cb}: {acc:.4f}")
