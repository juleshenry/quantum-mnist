"""
Phase Three: Generalized Multi-Class Quantum Classifier for MNIST (0-9)

This module implements a quantum classifier that generalizes the binary classifier
from Phase One to handle multi-class classification for all 10 digits (0-9).

Approach: Use multiple readout qubits (one per class) to enable multi-class classification.
Each readout qubit's expectation value serves as a logit for that class.
"""

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import collections
from typing import Tuple, List

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class CircuitLayerBuilder:
    """Helper class to build parameterized quantum circuit layers."""
    
    def __init__(self, data_qubits, readout_qubits):
        """
        Initialize the circuit layer builder.
        
        Args:
            data_qubits: List of qubits for encoding input data
            readout_qubits: List of readout qubits for multi-class output
        """
        self.data_qubits = data_qubits
        self.readout_qubits = readout_qubits
    
    def add_layer(self, circuit, gate, prefix):
        """
        Add a parameterized layer to the circuit.
        Each data qubit interacts with each readout qubit.
        
        Args:
            circuit: The quantum circuit to add layers to
            gate: The two-qubit gate to use (e.g., cirq.XX, cirq.ZZ)
            prefix: Prefix for parameter names
        """
        for j, readout in enumerate(self.readout_qubits):
            for i, data_qubit in enumerate(self.data_qubits):
                symbol = sympy.Symbol(f'{prefix}_r{j}_d{i}')
                circuit.append(gate(data_qubit, readout)**symbol)


def load_and_preprocess_mnist(num_classes: int = 10) -> Tuple:
    """
    Load and preprocess MNIST dataset for multi-class classification.
    
    Args:
        num_classes: Number of classes to include (default: 10 for all digits 0-9)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Rescale to [0, 1]
    x_train = x_train[..., np.newaxis] / 255.0
    x_test = x_test[..., np.newaxis] / 255.0
    
    # Filter to include all digits 0-9
    if num_classes < 10:
        train_filter = y_train < num_classes
        test_filter = y_test < num_classes
        x_train, y_train = x_train[train_filter], y_train[train_filter]
        x_test, y_test = x_test[test_filter], y_test[test_filter]
    
    print(f"Training examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")
    
    return x_train, y_train, x_test, y_test


def resize_and_binarize(x_train, x_test, image_size: Tuple[int, int] = (4, 4), 
                        threshold: float = 0.5) -> Tuple:
    """
    Resize images and convert to binary representation.
    
    Args:
        x_train: Training images
        x_test: Test images
        image_size: Target size for quantum encoding
        threshold: Binarization threshold
    
    Returns:
        Tuple of (x_train_bin, x_test_bin)
    """
    # Resize images
    x_train_small = tf.image.resize(x_train, image_size).numpy()
    x_test_small = tf.image.resize(x_test, image_size).numpy()
    
    # Binarize
    x_train_bin = np.array(x_train_small > threshold, dtype=np.float32)
    x_test_bin = np.array(x_test_small > threshold, dtype=np.float32)
    
    return x_train_bin, x_test_bin


def remove_contradicting(xs, ys):
    """
    Remove images that have contradictory labels after resizing and binarization.
    
    Args:
        xs: Input images
        ys: Labels
    
    Returns:
        Tuple of (filtered_xs, filtered_ys)
    """
    mapping = collections.defaultdict(set)
    orig_x = {}
    
    # Map each unique image to its set of labels
    for x, y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    
    # Keep only images with a single unique label
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
    
    num_removed = len(xs) - len(new_x)
    print(f"Removed {num_removed} contradicting examples")
    
    return np.array(new_x), np.array(new_y)


def convert_to_circuit(image, image_size: Tuple[int, int] = (4, 4)):
    """
    Encode a binary image into a quantum circuit.
    
    For each pixel with value 1, apply an X gate to the corresponding qubit.
    
    Args:
        image: Binary image array
        image_size: Size of the image (height, width), must match image dimensions
    
    Returns:
        cirq.Circuit encoding the image
    """
    values = image.flatten()
    qubits = cirq.GridQubit.rect(*image_size)
    circuit = cirq.Circuit()
    
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    
    return circuit


def create_quantum_model(num_classes: int = 10, 
                         image_size: Tuple[int, int] = (4, 4)) -> Tuple:
    """
    Create a multi-class quantum neural network model.
    
    Uses multiple readout qubits (one per class) to enable multi-class classification.
    
    Args:
        num_classes: Number of output classes
        image_size: Size of input images (height, width)
    
    Returns:
        Tuple of (circuit, readout_operators)
    """
    # Create data qubits for encoding input
    data_qubits = cirq.GridQubit.rect(*image_size)
    
    # Create readout qubits (one per class)
    readout_qubits = [cirq.GridQubit(-1, i) for i in range(num_classes)]
    
    circuit = cirq.Circuit()
    
    # Prepare readout qubits
    for readout in readout_qubits:
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))
    
    # Build parameterized layers
    builder = CircuitLayerBuilder(data_qubits=data_qubits, 
                                  readout_qubits=readout_qubits)
    
    # Add entangling layers
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    
    # Final readout preparation
    for readout in readout_qubits:
        circuit.append(cirq.H(readout))
    
    # Create readout operators (Z measurement on each readout qubit)
    readout_operators = [cirq.Z(readout) for readout in readout_qubits]
    
    return circuit, readout_operators


def create_classical_model(num_classes: int = 10, 
                           image_size: Tuple[int, int] = (4, 4)):
    """
    Create a simple classical neural network baseline for comparison.
    
    Note: This baseline uses fewer parameters than the quantum model.
    Quantum model: ~320 parameters (2 layers × 16 data × 10 readout qubits)
    Classical model: ~54 parameters ((16×2 + 2) + (2×10 + 10))
    
    Args:
        num_classes: Number of output classes
        image_size: Size of input images
    
    Returns:
        Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(*image_size, 1)),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    
    return model


def main():
    """Main function to train and evaluate the multi-class quantum classifier."""
    
    print("="*60)
    print("Phase Three: Multi-Class Quantum Classifier for MNIST (0-9)")
    print("="*60)
    
    # Configuration
    NUM_CLASSES = 10
    IMAGE_SIZE = (4, 4)
    THRESHOLD = 0.5
    BATCH_SIZE = 32
    EPOCHS = 3
    TRAIN_SIZE = 1000  # Use subset for faster training
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing MNIST data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist(NUM_CLASSES)
    
    # Step 2: Resize and binarize
    print("\n2. Resizing and binarizing images...")
    x_train_bin, x_test_bin = resize_and_binarize(x_train, x_test, 
                                                   IMAGE_SIZE, THRESHOLD)
    
    # Step 3: Remove contradictions
    print("\n3. Removing contradicting examples...")
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_bin, y_train)
    
    # Use subset for training
    if len(x_train_nocon) > TRAIN_SIZE:
        indices = np.random.choice(len(x_train_nocon), TRAIN_SIZE, replace=False)
        x_train_nocon = x_train_nocon[indices]
        y_train_nocon = y_train_nocon[indices]
        print(f"Using {TRAIN_SIZE} training examples")
    
    # Step 4: Convert to quantum circuits
    print("\n4. Converting images to quantum circuits...")
    x_train_circ = [convert_to_circuit(x, IMAGE_SIZE) for x in x_train_nocon]
    x_test_circ = [convert_to_circuit(x, IMAGE_SIZE) for x in x_test_bin]
    
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
    
    print(f"Created {len(x_train_circ)} training circuits")
    print(f"Created {len(x_test_circ)} test circuits")
    
    # Step 5: Create quantum model
    print("\n5. Building quantum neural network...")
    model_circuit, model_readout = create_quantum_model(NUM_CLASSES, IMAGE_SIZE)
    
    # Build Keras model with PQC layer
    quantum_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # PQC layer returns expectation values for each readout operator
        tfq.layers.PQC(model_circuit, model_readout),
    ])
    
    quantum_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
        metrics=['accuracy']
    )
    
    print("\nQuantum Model Summary:")
    print(f"Input: Quantum circuits (encoded images)")
    print(f"Output: {NUM_CLASSES} logits (one per class)")
    print(f"Readout qubits: {NUM_CLASSES}")
    
    # Step 6: Train quantum model
    print(f"\n6. Training quantum model for {EPOCHS} epochs...")
    qnn_history = quantum_model.fit(
        x_train_tfcirc, y_train_nocon,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test)
    )
    
    print("\n7. Evaluating quantum model...")
    qnn_results = quantum_model.evaluate(x_test_tfcirc, y_test)
    print(f"Quantum Model - Test Loss: {qnn_results[0]:.4f}, Test Accuracy: {qnn_results[1]:.4f}")
    
    # Step 8: Create and train classical baseline
    print("\n8. Training classical baseline for comparison...")
    classical_model = create_classical_model(NUM_CLASSES, IMAGE_SIZE)
    classical_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
        metrics=['accuracy']
    )
    
    classical_history = classical_model.fit(
        x_train_nocon, y_train_nocon,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test_bin, y_test)
    )
    
    print("\n9. Evaluating classical model...")
    classical_results = classical_model.evaluate(x_test_bin, y_test)
    print(f"Classical Model - Test Loss: {classical_results[0]:.4f}, Test Accuracy: {classical_results[1]:.4f}")
    
    # Step 10: Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Quantum Model:   Accuracy = {qnn_results[1]*100:.2f}%")
    print(f"Classical Model: Accuracy = {classical_results[1]*100:.2f}%")
    print("\nNote: This demonstrates the generalization from binary to multi-class")
    print("quantum classification. The architecture uses one readout qubit per class,")
    print("enabling true multi-class quantum classification for digits 0-9.")
    print("="*60)


if __name__ == "__main__":
    main()
