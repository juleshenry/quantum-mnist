# quantum_image_classifier.py
import os
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import random

class QuantumImageClassifier:
    """
    Quantum Machine Learning Classifier for Binary Image Classification
    Uses TensorFlow Quantum and Cirq for hybrid quantum-classical computing
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (32, 32),
                 n_qubits: int = 8,
                 n_layers: int = 2,
                 learning_rate: float = 0.001):
        """
        Initialize the quantum classifier
        
        Args:
            image_size: Size to resize images to (height, width)
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of variational layers
            learning_rate: Learning rate for optimizer
        """
        self.image_size = image_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.model = None
        self.qubits = cirq.GridQubit.rect(1, n_qubits)  # Line of qubits
        
    def load_and_preprocess_images(self, 
                                   class_0_dir: str, 
                                   class_1_dir: str,
                                   test_size: float = 0.2,
                                   random_state: int = 42) -> Tuple:
        """
        Load images from two directories and preprocess them
        
        Args:
            class_0_dir: Directory containing images for class 0
            class_1_dir: Directory containing images for class 1
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data, train_labels, test_labels)
        """
        def load_images_from_dir(directory: str, label: int) -> List:
            images = []
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            
            for filename in os.listdir(directory):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(directory, filename)
                    try:
                        # Load and preprocess image
                        img = Image.open(img_path).convert('L')  # Convert to grayscale
                        img = img.resize(self.image_size)
                        img_array = np.array(img) / 255.0  # Normalize to [0,1]
                        images.append(img_array.flatten())  # Flatten the image
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            return images
        
        print(f"Loading images from {class_0_dir} and {class_1_dir}...")
        
        # Load images from both classes
        class_0_images = load_images_from_dir(class_0_dir, 0)
        class_1_images = load_images_from_dir(class_1_dir, 1)
        
        print(f"Loaded {len(class_0_images)} images for class 0")
        print(f"Loaded {len(class_1_images)} images for class 1")
        
        # Create labels
        labels_0 = [0] * len(class_0_images)
        labels_1 = [1] * len(class_1_images)
        
        # Combine data
        X = np.array(class_0_images + class_1_images)
        y = np.array(labels_0 + labels_1)
        
        # Shuffle and split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_for_quantum(self, X: np.ndarray) -> List:
        """
        Preprocess classical data for quantum circuit input
        
        Args:
            X: Input data array
            
        Returns:
            List of quantum circuits for data encoding
        """
        def encode_image_to_circuit(image_data: np.ndarray) -> cirq.Circuit:
            """Encode image data into quantum circuit using amplitude encoding"""
            circuit = cirq.Circuit()
            
            # Normalize image data
            normalized_data = image_data / np.linalg.norm(image_data)
            
            # Downsample to match number of qubits
            if len(normalized_data) > self.n_qubits:
                # Take first n_qubits features
                features = normalized_data[:self.n_qubits]
            else:
                # Pad if necessary
                features = np.pad(normalized_data, 
                                (0, self.n_qubits - len(normalized_data)), 
                                mode='constant')
            
            # Apply rotations based on pixel values
            for i, qubit in enumerate(self.qubits):
                angle = np.arccos(features[i]) * 2  # Map to rotation angle
                circuit.append(cirq.ry(angle)(qubit))
            
            return circuit
        
        return [encode_image_to_circuit(x) for x in X]
    
    def create_quantum_model(self) -> tf.keras.Model:
        """
        Create hybrid quantum-classical model
        
        Returns:
            TensorFlow Keras model
        """
        # Create parameterized quantum circuit
        circuit = cirq.Circuit()
        
        # Create symbolic parameters
        params = sympy.symbols(f'theta_0:{self.n_qubits * self.n_layers}')
        
        # Build variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for i, qubit in enumerate(self.qubits):
                circuit.append(cirq.ry(params[param_idx])(qubit))
                param_idx += 1
            
            # Entangling layer (nearest-neighbor CNOTs)
            for i in range(self.n_qubits - 1):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
        
        # Add measurement
        circuit.append(cirq.measure(*self.qubits, key='measurement'))
        
        # Create input layers
        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)
        control_params = tf.keras.Input(shape=(self.n_qubits * self.n_layers,))
        
        # PQC layer
        pqc_layer = tfq.layers.PQC(
            circuit,
            cirq.Z(self.qubits[0])  # Measure expectation value on first qubit
        )
        
        expectation = pqc_layer(circuit_input, symbol_values=control_params)
        
        # Classical post-processing
        dense1 = tf.keras.layers.Dense(16, activation='relu')(expectation)
        dense2 = tf.keras.layers.Dense(8, activation='relu')(dense1)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
        
        # Create model
        model = tf.keras.Model(
            inputs=[circuit_input, control_params],
            outputs=output
        )
        
        return model
    
    def build_model(self):
        """Build and compile the quantum model"""
        self.model = self.create_quantum_model()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print(self.model.summary())
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: np.ndarray = None,
             y_val: np.ndarray = None,
             epochs: int = 50,
             batch_size: int = 16,
             validation_split: float = 0.1):
        """
        Train the quantum model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of training data to use for validation
        """
        # Convert data to quantum circuits
        print("Converting training data to quantum circuits...")
        X_train_circuits = self.preprocess_for_quantum(X_train)
        
        # Create random initial parameters
        n_params = self.n_qubits * self.n_layers
        param_values = np.random.uniform(-np.pi, np.pi, size=(len(X_train_circuits), n_params))
        
        if X_val is not None:
            print("Converting validation data to quantum circuits...")
            X_val_circuits = self.preprocess_for_quantum(X_val)
            param_values_val = np.random.uniform(-np.pi, np.pi, size=(len(X_val_circuits), n_params))
            validation_data = ([X_val_circuits, param_values_val], y_val)
        else:
            validation_split = validation_split
            validation_data = None
        
        # Train the model
        history = self.model.fit(
            x=[X_train_circuits, param_values],
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Converting test data to quantum circuits...")
        X_test_circuits = self.preprocess_for_quantum(X_test)
        
        n_params = self.n_qubits * self.n_layers
        param_values_test = np.random.uniform(-np.pi, np.pi, size=(len(X_test_circuits), n_params))
        
        # Make predictions
        predictions = self.model.predict([X_test_circuits, param_values_test])
        predictions_binary = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        results = {
            'classification_report': classification_report(y_test, predictions_binary),
            'confusion_matrix': confusion_matrix(y_test, predictions_binary),
            'accuracy': np.mean(predictions_binary.flatten() == y_test)
        }
        
        return results
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names=['Class 0', 'Class 1']):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()

def create_sample_data(n_samples: int = 80, image_size: Tuple[int, int] = (32, 32)):
    """
    Create sample data for testing when real images aren't available
    
    Args:
        n_samples: Number of samples per class
        image_size: Size of images to generate
        
    Returns:
        Tuple of (class_0_data, class_1_data)
    """
    np.random.seed(42)
    
    # Create synthetic data with different patterns for two classes
    n_pixels = image_size[0] * image_size[1]
    
    # Class 0: vertical stripe pattern
    class_0_data = []
    for i in range(n_samples):
        img = np.zeros((image_size[0], image_size[1]))
        stripe_width = 4
        for j in range(0, image_size[1], stripe_width*2):
            img[:, j:j+stripe_width] = np.random.uniform(0.7, 1.0)
        img += np.random.normal(0, 0.1, img.shape)  # Add noise
        class_0_data.append(img.flatten())
    
    # Class 1: horizontal stripe pattern
    class_1_data = []
    for i in range(n_samples):
        img = np.zeros((image_size[0], image_size[1]))
        stripe_height = 4
        for j in range(0, image_size[0], stripe_height*2):
            img[j:j+stripe_height, :] = np.random.uniform(0.7, 1.0)
        img += np.random.normal(0, 0.1, img.shape)  # Add noise
        class_1_data.append(img.flatten())
    
    return np.array(class_0_data), np.array(class_1_data)

def main():
    """Main execution function"""
    
    # Configuration
    IMAGE_SIZE = (32, 32)
    N_QUBITS = 8
    N_LAYERS = 2
    EPOCHS = 50
    BATCH_SIZE = 16
    
    # Initialize classifier
    classifier = QuantumImageClassifier(
        image_size=IMAGE_SIZE,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS
    )
    
    # Check if directories exist, otherwise use sample data
    class_0_dir = "./data/class_0"  # Replace with your path
    class_1_dir = "./data/class_1"  # Replace with your path
    
    if os.path.exists(class_0_dir) and os.path.exists(class_1_dir):
        print("Loading real images from directories...")
        # Load real images
        X_train, X_test, y_train, y_test = classifier.load_and_preprocess_images(
            class_0_dir, class_1_dir
        )
    else:
        print("Directories not found. Using synthetic sample data...")
        print("(Replace with your actual image directories for real data)")
        
        # Create sample data
        class_0_data, class_1_data = create_sample_data(n_samples=80, image_size=IMAGE_SIZE)
        
        # Combine and split
        X = np.vstack([class_0_data, class_1_data])
        y = np.hstack([np.zeros(80), np.ones(80)])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Build the quantum model
    print("\nBuilding quantum model...")
    classifier.build_model()
    
    # Train the model
    print("\nTraining quantum model...")
    history = classifier.train(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    results = classifier.evaluate(X_test, y_test)
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    print("\nTraining complete! Check the generated plots for results.")

if __name__ == "__main__":
    main()