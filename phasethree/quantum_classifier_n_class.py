"""
Phase Three: Generalized N-class Quantum Classifier
This module extends the binary quantum classifier to support multi-class classification.
"""

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import collections


def filter_classes(x, y, classes):
    """Filter dataset to only include specified classes."""
    keep = np.isin(y, classes)
    x, y = x[keep], y[keep]
    return x, y


def resize_images(x, size=(4, 4)):
    """Resize images to specified size."""
    return tf.image.resize(x, size).numpy()


def remove_contradicting(xs, ys):
    """Remove images that have contradicting labels."""
    mapping = collections.defaultdict(set)
    orig_x = {}
    
    # Determine the set of labels for each unique image
    for x, y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
        else:
            # Throw out images that match more than one label
            pass
    
    print(f"Number of unique images: {len(mapping)}")
    print(f"Initial number of images: {len(xs)}")
    print(f"Remaining non-contradicting unique images: {len(new_x)}")
    
    return np.array(new_x), np.array(new_y)


def binarize_images(x, threshold=0.5):
    """Convert images to binary (0 or 1) based on threshold."""
    return np.array(x > threshold, dtype=np.float32)


def convert_to_circuit(image):
    """Encode truncated classical image into quantum circuit."""
    values = image.flatten()
    h, w = image.shape[0], image.shape[1]
    qubits = cirq.GridQubit.rect(h, w)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


class CircuitLayerBuilder:
    """Helper class to build quantum circuit layers."""
    
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def create_quantum_model(image_size=(4, 4)):
    """Create a quantum neural network model circuit and readout operation."""
    h, w = image_size
    data_qubits = cirq.GridQubit.rect(h, w)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()
    
    # Prepare the readout qubit
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout
    )
    
    # Add layers
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    
    # Prepare the readout qubit
    circuit.append(cirq.H(readout))
    
    return circuit, cirq.Z(readout)


def create_multiclass_quantum_model(num_classes, image_size=(4, 4)):
    """
    Create a multi-class quantum classifier using one-vs-rest approach.
    Returns a list of quantum circuits and readouts, one for each class.
    """
    circuits = []
    readouts = []
    
    for i in range(num_classes):
        circuit, readout = create_quantum_model(image_size)
        circuits.append(circuit)
        readouts.append(readout)
    
    return circuits, readouts


def build_quantum_model(num_classes, image_size=(4, 4)):
    """
    Build a Keras model with multiple PQC layers for multi-class classification.
    Uses one-vs-rest approach with separate quantum circuits per class.
    """
    circuits, readouts = create_multiclass_quantum_model(num_classes, image_size)
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
    
    # Create PQC layer for each class
    pqc_outputs = []
    for i in range(num_classes):
        pqc = tfq.layers.PQC(circuits[i], readouts[i])
        pqc_outputs.append(pqc(inputs))
    
    # Concatenate outputs from all PQC layers
    if num_classes > 1:
        concatenated = tf.keras.layers.Concatenate()(pqc_outputs)
    else:
        concatenated = pqc_outputs[0]
    
    # Softmax activation for multi-class classification
    outputs = tf.keras.layers.Activation('softmax')(concatenated)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def preprocess_data(x_train, y_train, x_test, y_test, image_size=(4, 4), threshold=0.5):
    """Complete preprocessing pipeline for quantum classification."""
    
    # Add channel dimension if needed
    if len(x_train.shape) == 3:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    
    # Rescale to [0, 1] if needed
    if x_train.max() > 1.0:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    
    # Resize images
    x_train_small = resize_images(x_train, image_size)
    x_test_small = resize_images(x_test, image_size)
    
    # Remove contradicting examples
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    
    # Binarize images
    x_train_bin = binarize_images(x_train_nocon, threshold)
    x_test_bin = binarize_images(x_test_small, threshold)
    
    # Remove contradicting binary examples
    x_train_bin, y_train_bin = remove_contradicting(x_train_bin, y_train_nocon)
    
    # Convert to quantum circuits
    x_train_circ = [convert_to_circuit(x[:, :, 0]) for x in x_train_bin]
    x_test_circ = [convert_to_circuit(x[:, :, 0]) for x in x_test_bin]
    
    # Convert to TensorFlow quantum tensors
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
    
    return x_train_tfcirc, y_train_bin, x_test_tfcirc, y_test, x_train_bin, x_test_bin


def train_quantum_classifier(x_train, y_train, x_test, y_test, 
                            num_classes=10, image_size=(4, 4),
                            epochs=3, batch_size=32):
    """
    Train a quantum classifier on the provided data.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        num_classes: Number of classes
        image_size: Size to resize images to
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        model: Trained Keras model
        history: Training history
        results: Evaluation results
    """
    
    # Preprocess data
    x_train_tfcirc, y_train_processed, x_test_tfcirc, y_test_processed, _, _ = \
        preprocess_data(x_train, y_train, x_test, y_test, image_size)
    
    # Convert labels to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train_processed, num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test_processed, num_classes)
    
    # Build model
    model = build_quantum_model(num_classes, image_size)
    
    # Compile model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train model
    history = model.fit(
        x_train_tfcirc, y_train_onehot,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test_onehot)
    )
    
    # Evaluate model
    results = model.evaluate(x_test_tfcirc, y_test_onehot)
    
    return model, history, results


if __name__ == "__main__":
    # Example: Test on MNIST digits 0-9
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # For testing, use a subset of classes (e.g., 0-4)
    print("Filtering to classes 0-4 for testing...")
    classes = [0, 1, 2, 3, 4]
    x_train, y_train = filter_classes(x_train, y_train, classes)
    x_test, y_test = filter_classes(x_test, y_test, classes)
    
    print(f"Training examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")
    
    # Train quantum classifier
    print("\nTraining quantum classifier...")
    model, history, results = train_quantum_classifier(
        x_train, y_train, x_test, y_test,
        num_classes=len(classes),
        image_size=(4, 4),
        epochs=3,
        batch_size=32
    )
    
    print(f"\nFinal test accuracy: {results[1]:.4f}")
