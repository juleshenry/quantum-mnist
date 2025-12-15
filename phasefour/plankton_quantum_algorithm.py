"""
Quantum Image Classification for Plankton Dataset

This module implements a quantum neural network for binary classification of plankton images.
It extends the quantum MNIST approach to handle the more complex plankton dataset.

Key features:
- Supports 8x8 and 16x16 image dimensions for quantum encoding
- Uses Cirq for quantum circuit construction
- Implements hybrid quantum-classical model with TensorFlow Quantum
- Handles grayscale conversion and normalization of plankton images
- Performs binary classification between plankton species pairs

Architecture:
- Quantum data encoding: Binary threshold encoding of grayscale pixels
- Quantum circuit: Parameterized quantum circuit (PQC) with XX and ZZ gates
- Classical post-processing: Dense layers for final classification
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import math

# Quantum and ML imports (to be installed: cirq, tensorflow, tensorflow-quantum, sympy)
try:
    import cirq
    import sympy
    import tensorflow as tf
    import tensorflow_quantum as tfq
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Please install: pip install cirq tensorflow tensorflow-quantum sympy")


class PlanktonQuantumClassifier:
    """
    Quantum classifier for plankton images using parameterized quantum circuits.
    """
    
    def __init__(self, image_size=(8, 8), seed=0, threshold=0.5):
        """
        Initialize the quantum classifier.
        
        Args:
            image_size: Tuple (height, width) for image dimensions. 
                       Recommended: (8, 8) for faster simulation, (16, 16) for better accuracy
            seed: Random seed for reproducibility
            threshold: Binary threshold for quantum encoding (default: 0.5)
                      Pixels > threshold map to |1⟩, others to |0⟩
        """
        self.image_size = image_size
        self.n_qubits = image_size[0] * image_size[1]
        self.threshold = threshold
        np.random.seed(seed)
        
        # Verify image size is feasible for quantum simulation
        if self.n_qubits > 64:
            print(f"Warning: {self.n_qubits} qubits may be too large for efficient simulation")
        
        print(f"Initialized quantum classifier with {image_size} images ({self.n_qubits} qubits)")
        print(f"Binary encoding threshold: {threshold}")
    
    def bilinear_interpolation(self, image, y, x):
        """Perform bilinear interpolation for smooth image resizing."""
        height = image.shape[0]
        width = image.shape[1]

        x1 = max(min(math.floor(x), width - 1), 0)
        y1 = max(min(math.floor(y), height - 1), 0)
        x2 = max(min(math.ceil(x), width - 1), 0)
        y2 = max(min(math.ceil(y), height - 1), 0)

        a = float(image[y1, x1])
        b = float(image[y2, x1])
        c = float(image[y1, x2])
        d = float(image[y2, x2])

        dx = x - x1
        dy = y - y1

        new_pixel = a * (1 - dx) * (1 - dy)
        new_pixel += b * dy * (1 - dx)
        new_pixel += c * dx * (1 - dy)
        new_pixel += d * dx * dy
        return round(new_pixel)

    def resize_image(self, image, new_height, new_width):
        """Resize image using bilinear interpolation."""
        new_image = np.zeros((new_height, new_width), image.dtype)
        
        orig_height = image.shape[0]
        orig_width = image.shape[1]

        # Compute center column and center row
        x_orig_center = (orig_width - 1) / 2
        y_orig_center = (orig_height - 1) / 2

        # Compute center of resized image
        x_scaled_center = (new_width - 1) / 2
        y_scaled_center = (new_height - 1) / 2

        # Compute the scale in both axes
        scale_x = orig_width / new_width
        scale_y = orig_height / new_height

        for y in range(new_height):
            for x in range(new_width):
                x_ = (x - x_scaled_center) * scale_x + x_orig_center
                y_ = (y - y_scaled_center) * scale_y + y_orig_center
                new_image[y, x] = self.bilinear_interpolation(image, y_, x_)
        return new_image

    def preprocess_image(self, image_path_or_obj):
        """
        Preprocess plankton image for quantum encoding.
        
        Args:
            image_path_or_obj: Path to image file or PIL Image object
            
        Returns:
            Normalized numpy array of shape image_size
        """
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj)
        else:
            image = image_path_or_obj
        
        # Convert to grayscale
        image = image.convert("L")
        
        # Convert to numpy array
        image_array = np.asarray(image)
        
        # Resize to target dimensions
        image_array = self.resize_image(image_array, *self.image_size)
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        
        return image_array

    def convert_to_circuit(self, image):
        """
        Encode preprocessed image into quantum circuit using binary threshold encoding.
        
        Args:
            image: Normalized numpy array of shape image_size
            
        Returns:
            Cirq Circuit encoding the image
        """
        # Flatten image to 1D array
        values = np.ndarray.flatten(image)
        
        # Create qubit grid matching image dimensions
        qubits = cirq.GridQubit.rect(*self.image_size)
        circuit = cirq.Circuit()
        
        # Binary encoding: apply X gate if pixel value > threshold
        for i, value in enumerate(values):
            if value > self.threshold:
                circuit.append(cirq.X(qubits[i]))
        
        return circuit

    def create_quantum_model(self):
        """
        Create parameterized quantum circuit for classification.
        
        Returns:
            Tuple of (circuit, readout_operator)
        """
        # Create data qubits in grid matching image dimensions
        data_qubits = cirq.GridQubit.rect(*self.image_size)
        
        # Single readout qubit
        readout = cirq.GridQubit(-1, -1)
        
        circuit = cirq.Circuit()
        
        # Prepare the readout qubit in superposition
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))
        
        # Build parameterized circuit layers
        builder = CircuitLayerBuilder(
            data_qubits=data_qubits,
            readout=readout
        )
        
        # Add entangling layers with parameterized gates
        # More layers = more expressivity but harder to train
        builder.add_layer(circuit, cirq.XX, "xx1")
        builder.add_layer(circuit, cirq.ZZ, "zz1")
        
        # For larger images, add more layers for better expressivity
        if self.n_qubits >= 64:
            builder.add_layer(circuit, cirq.XX, "xx2")
            builder.add_layer(circuit, cirq.ZZ, "zz2")
        
        # Final readout preparation
        circuit.append(cirq.H(readout))
        
        return circuit, cirq.Z(readout)

    def build_model(self):
        """
        Build hybrid quantum-classical Keras model.
        
        Returns:
            Compiled Keras model
        """
        # Create quantum circuit and readout
        model_circuit, model_readout = self.create_quantum_model()
        
        # Build hybrid model
        model = tf.keras.Sequential([
            # Input: quantum circuits encoded as strings
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # Quantum layer: PQC returns expected value of readout operator
            tfq.layers.PQC(model_circuit, model_readout),
            # Classical post-processing
            tf.keras.layers.Dense(1)
        ])
        
        return model

    def load_plankton_images(self, plankton_dir, category, max_images=100):
        """
        Load plankton images from local directory.
        
        Args:
            plankton_dir: Base directory containing plankton categories
            category: Category name (subfolder)
            max_images: Maximum number of images to load
            
        Returns:
            List of preprocessed images
        """
        category_path = os.path.join(plankton_dir, category, 'training_data')
        
        if not os.path.exists(category_path):
            raise ValueError(f"Category path does not exist: {category_path}")
        
        image_files = [f for f in os.listdir(category_path) if f.endswith('.jpeg')]
        image_files = image_files[:max_images]
        
        images = []
        for img_file in image_files:
            img_path = os.path.join(category_path, img_file)
            try:
                preprocessed = self.preprocess_image(img_path)
                images.append(preprocessed)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
        
        print(f"Loaded {len(images)} images from {category}")
        return images

    def prepare_binary_dataset(self, images_a, images_b, train_ratio=0.75):
        """
        Prepare binary classification dataset from two categories.
        
        Args:
            images_a: List of images for class 0
            images_b: List of images for class 1
            train_ratio: Fraction of data for training
            
        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        # Split into train/test
        split_a = int(len(images_a) * train_ratio)
        split_b = int(len(images_b) * train_ratio)
        
        x_train = images_a[:split_a] + images_b[:split_b]
        x_test = images_a[split_a:] + images_b[split_b:]
        
        y_train = np.array([0] * split_a + [1] * split_b)
        y_test = np.array([0] * (len(images_a) - split_a) + [1] * (len(images_b) - split_b))
        
        # Shuffle training data
        shuffle_idx = np.random.permutation(len(x_train))
        x_train = [x_train[i] for i in shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        # Shuffle test data
        shuffle_idx = np.random.permutation(len(x_test))
        x_test = [x_test[i] for i in shuffle_idx]
        y_test = y_test[shuffle_idx]
        
        return np.array(x_train), y_train, np.array(x_test), y_test

    def train_binary_classifier(self, category_a, category_b, 
                                plankton_dir='data/zooplankton_0p5x',
                                max_images=50, epochs=20, batch_size=16):
        """
        Train quantum classifier on binary classification task.
        
        Args:
            category_a: First plankton category name
            category_b: Second plankton category name
            plankton_dir: Base directory for plankton data
            max_images: Maximum images per category
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Tuple of (model, history, test_accuracy)
        """
        print(f"\n{'='*60}")
        print(f"Training: {category_a} vs {category_b}")
        print(f"{'='*60}\n")
        
        # Load images
        images_a = self.load_plankton_images(plankton_dir, category_a, max_images)
        images_b = self.load_plankton_images(plankton_dir, category_b, max_images)
        
        # Prepare dataset
        x_train, y_train, x_test, y_test = self.prepare_binary_dataset(
            images_a, images_b
        )
        
        # Convert to quantum circuits
        print("Converting images to quantum circuits...")
        x_train_circ = [self.convert_to_circuit(x) for x in x_train]
        x_test_circ = [self.convert_to_circuit(x) for x in x_test]
        
        # Convert to TensorFlow Quantum format
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        
        # Build and compile model
        print("Building quantum model...")
        model = self.build_model()
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            metrics=['accuracy']
        )
        
        # Train
        print(f"Training for {epochs} epochs...")
        history = model.fit(
            x_train_tfcirc, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test_tfcirc, y_test)
        )
        
        # Evaluate
        results = model.evaluate(x_test_tfcirc, y_test, return_dict=True)
        print(f"\nTest Loss: {results['loss']:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        
        return model, history, results['accuracy']


class CircuitLayerBuilder:
    """
    Helper class to build parameterized quantum circuit layers.
    Each layer applies a parameterized 2-qubit gate between each data qubit and the readout.
    """
    
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        """
        Add a parameterized layer to the circuit.
        
        Args:
            circuit: Cirq Circuit to add layer to
            gate: 2-qubit gate (e.g., cirq.XX, cirq.ZZ)
            prefix: Prefix for parameter symbols
        """
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def run_cartesian_comparison(plankton_dir='data/zooplankton_0p5x',
                             image_size=(8, 8),
                             max_pairs=5,
                             max_images=30,
                             epochs=15):
    """
    Run quantum classification on multiple pairs of plankton categories.
    
    This function demonstrates the "fair" comparison approach used in the classical case,
    applied to quantum circuits.
    
    Args:
        plankton_dir: Directory containing plankton data
        image_size: Tuple of (height, width) for quantum encoding
        max_pairs: Maximum number of category pairs to test
        max_images: Maximum images per category
        epochs: Number of training epochs per pair
    """
    # List of plankton categories
    plankton_categories = [
        "aphanizomenon", "asplanchna", "asterionella", "bosmina", "brachionus",
        "ceratium", "chaoborus", "conochilus", "cyclops", "daphnia"
    ]
    
    # Initialize classifier
    classifier = PlanktonQuantumClassifier(image_size=image_size)
    
    # Generate category pairs (cartesian product excluding self-pairs)
    pairs = [(a, b) for a in plankton_categories for b in plankton_categories if a < b]
    pairs = pairs[:max_pairs]
    
    results = []
    
    print(f"\nRunning quantum classification on {len(pairs)} category pairs")
    print(f"Image size: {image_size}, Max images: {max_images}, Epochs: {epochs}\n")
    
    for cat_a, cat_b in pairs:
        try:
            _, _, accuracy = classifier.train_binary_classifier(
                cat_a, cat_b,
                plankton_dir=plankton_dir,
                max_images=max_images,
                epochs=epochs,
                batch_size=8
            )
            results.append({
                'category_a': cat_a,
                'category_b': cat_b,
                'accuracy': accuracy
            })
        except Exception as e:
            print(f"Error training {cat_a} vs {cat_b}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("QUANTUM CLASSIFICATION RESULTS")
    print("="*60)
    for result in results:
        print(f"{result['category_a']:20s} vs {result['category_b']:20s}: {result['accuracy']:.4f}")
    
    if results:
        mean_accuracy = np.mean([r['accuracy'] for r in results])
        print(f"\nMean Accuracy: {mean_accuracy:.4f}")
    
    return results


if __name__ == "__main__":
    """
    Example usage: Train quantum classifier on plankton data.
    
    This demonstrates quantum image classification on the plankton dataset,
    extending the approach from 4x4 MNIST to larger, more complex images.
    """
    
    # Use 8x8 for faster training/testing (64 qubits)
    # Use 16x16 for better accuracy but slower simulation (256 qubits - may be too large!)
    
    print("Quantum Plankton Classification Demo")
    print("=" * 60)
    
    # Determine data directory relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data", "zooplankton_0p5x")
    
    # Quick test with single pair
    classifier = PlanktonQuantumClassifier(image_size=(8, 8))
    
    try:
        model, history, accuracy = classifier.train_binary_classifier(
            category_a="bosmina",
            category_b="cyclops",
            plankton_dir=data_dir,
            max_images=20,
            epochs=10,
            batch_size=4
        )
        
        print("\n" + "="*60)
        print("Single pair test completed successfully!")
        print(f"Final test accuracy: {accuracy:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nThis is expected if TensorFlow Quantum is not installed.")
        print("To run this code, install: pip install tensorflow tensorflow-quantum cirq")
