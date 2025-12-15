"""
Phase Four: Apply Quantum Classifier to Plankton Dataset
This module applies the generalized N-class quantum classifier to the plankton dataset
and compares results with deep learning approaches.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import phase three module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phasethree.quantum_classifier_n_class import (
    train_quantum_classifier,
    preprocess_data
)


# Plankton class names (35 classes)
PLANKTON_CLASSES = [
    "aphanizomenon", "asplanchna", "asterionella", "bosmina", "brachionus",
    "ceratium", "chaoborus", "conochilus", "copepod_skins", "cyclops",
    "daphnia", "daphnia_skins", "diaphanosoma", "diatom_chain", "dinobryon",
    "dirt", "eudiaptomus", "filament", "fish", "fragilaria",
    "hydra", "kellicottia", "keratella_cochlearis", "keratella_quadrata",
    "leptodora", "maybe_cyano", "nauplius", "paradileptus", "polyarthra",
    "rotifers", "synchaeta", "trichocerca", "unknown", "unknown_plankton",
    "uroglena"
]


def load_plankton_images(data_dir, max_images_per_class=100, subset_classes=None):
    """
    Load plankton images from the dataset directory.
    
    Args:
        data_dir: Path to the zooplankton_0p5x directory
        max_images_per_class: Maximum number of images to load per class
        subset_classes: List of class names to load (None for all)
        
    Returns:
        images: numpy array of images
        labels: numpy array of labels (class indices)
        class_names: list of class names
    """
    
    if subset_classes is None:
        classes = PLANKTON_CLASSES
    else:
        classes = subset_classes
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name, 'training_data')
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        # Get list of image files
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
        
        # Limit number of images per class
        image_files = image_files[:max_images_per_class]
        
        print(f"Loading {len(image_files)} images from class '{class_name}'...")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load image and convert to grayscale
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Classes: {len(classes)}")
    
    return np.array(images), np.array(labels), classes


def split_train_test(images, labels, test_ratio=0.2):
    """Split data into training and test sets."""
    # Shuffle the data
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    # Split
    split_idx = int(len(images) * (1 - test_ratio))
    x_train = images[:split_idx]
    y_train = labels[:split_idx]
    x_test = images[split_idx:]
    y_test = labels[split_idx:]
    
    return x_train, y_train, x_test, y_test


def create_classical_cnn_model(num_classes, input_shape=(64, 64, 1)):
    """
    Create a classical CNN model for comparison.
    Based on the approach from the plankton paper.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_fair_classical_model(num_classes, input_shape=(4, 4, 1)):
    """
    Create a fair classical model with similar complexity to quantum model.
    Uses 4x4 input like the quantum model for fair comparison.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_classical_model(x_train, y_train, x_test, y_test, 
                         num_classes, model_type='cnn', 
                         epochs=10, batch_size=32):
    """
    Train a classical neural network on the plankton data.
    
    Args:
        x_train, y_train: Training data
        x_test, y_test: Test data
        num_classes: Number of classes
        model_type: 'cnn' for full CNN or 'fair' for fair comparison
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        model: Trained model
        history: Training history
        results: Evaluation results
    """
    
    # Normalize images
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0
    
    # Add channel dimension if needed
    if len(x_train_norm.shape) == 3:
        x_train_norm = x_train_norm[..., np.newaxis]
        x_test_norm = x_test_norm[..., np.newaxis]
    
    # Resize for fair comparison if needed
    if model_type == 'fair':
        x_train_resized = tf.image.resize(x_train_norm, (4, 4)).numpy()
        x_test_resized = tf.image.resize(x_test_norm, (4, 4)).numpy()
        model = create_fair_classical_model(num_classes)
    else:
        # Resize to standard size for CNN
        x_train_resized = tf.image.resize(x_train_norm, (64, 64)).numpy()
        x_test_resized = tf.image.resize(x_test_norm, (64, 64)).numpy()
        model = create_classical_cnn_model(num_classes, input_shape=(64, 64, 1))
    
    # Convert labels to one-hot
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train model
    history = model.fit(
        x_train_resized, y_train_onehot,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test_resized, y_test_onehot)
    )
    
    # Evaluate
    results = model.evaluate(x_test_resized, y_test_onehot)
    
    return model, history, results


def plot_comparison(quantum_acc, cnn_acc, fair_nn_acc, class_names):
    """Plot comparison of different models."""
    models = ['Quantum\n(4x4)', 'Classical CNN\n(64x64)', 'Fair Classical\n(4x4)']
    accuracies = [quantum_acc, cnn_acc, fair_nn_acc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
    plt.ylabel('Accuracy')
    plt.title(f'Plankton Classification Accuracy Comparison\n({len(class_names)} classes)')
    plt.ylim([0, 1.0])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plankton_comparison.png', dpi=150)
    print("Comparison plot saved to plankton_comparison.png")
    

def main():
    """Main function to run the complete plankton quantum classification experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configuration
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'data', 'zooplankton_0p5x')
    
    # Use a subset of classes for feasibility (quantum simulation is expensive)
    # Start with 5 classes as a proof of concept
    SUBSET_CLASSES = ["daphnia", "cyclops", "bosmina", "diaphanosoma", "copepod_skins"]
    MAX_IMAGES_PER_CLASS = 50  # Limit images per class for faster training
    
    print("="*60)
    print("Phase Four: Quantum Classification of Plankton Dataset")
    print("="*60)
    
    # Load data
    print("\n1. Loading plankton images...")
    images, labels, class_names = load_plankton_images(
        DATA_DIR, 
        max_images_per_class=MAX_IMAGES_PER_CLASS,
        subset_classes=SUBSET_CLASSES
    )
    
    if len(images) == 0:
        print("Error: No images loaded. Please check the data directory.")
        return
    
    # Split data
    print("\n2. Splitting data into train/test sets...")
    x_train, y_train, x_test, y_test = split_train_test(images, labels, test_ratio=0.2)
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    num_classes = len(class_names)
    
    # Train quantum classifier
    print("\n3. Training Quantum Classifier...")
    print("(This may take a while due to quantum simulation overhead)")
    try:
        quantum_model, quantum_history, quantum_results = train_quantum_classifier(
            x_train, y_train, x_test, y_test,
            num_classes=num_classes,
            image_size=(4, 4),
            epochs=3,
            batch_size=16  # Smaller batch size for quantum
        )
        quantum_accuracy = quantum_results[1]
        print(f"Quantum Model Test Accuracy: {quantum_accuracy:.4f}")
    except Exception as e:
        print(f"Error training quantum model: {e}")
        quantum_accuracy = 0.0
    
    # Train fair classical model (4x4 input for fair comparison)
    print("\n4. Training Fair Classical Model (4x4 input)...")
    fair_model, fair_history, fair_results = train_classical_model(
        x_train, y_train, x_test, y_test,
        num_classes=num_classes,
        model_type='fair',
        epochs=20,
        batch_size=32
    )
    fair_accuracy = fair_results[1]
    print(f"Fair Classical Model Test Accuracy: {fair_accuracy:.4f}")
    
    # Train full CNN model (64x64 input, deeper network)
    print("\n5. Training Deep Learning CNN (64x64 input)...")
    cnn_model, cnn_history, cnn_results = train_classical_model(
        x_train, y_train, x_test, y_test,
        num_classes=num_classes,
        model_type='cnn',
        epochs=10,
        batch_size=32
    )
    cnn_accuracy = cnn_results[1]
    print(f"CNN Model Test Accuracy: {cnn_accuracy:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {len(class_names)} plankton classes")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print()
    print(f"Quantum Classifier (4x4):        {quantum_accuracy:.4f}")
    print(f"Fair Classical NN (4x4):         {fair_accuracy:.4f}")
    print(f"Deep Learning CNN (64x64):       {cnn_accuracy:.4f}")
    print("="*60)
    
    # Plot comparison
    print("\n6. Generating comparison plot...")
    plot_comparison(quantum_accuracy, cnn_accuracy, fair_accuracy, class_names)
    
    print("\nExperiment complete!")
    

if __name__ == "__main__":
    main()
