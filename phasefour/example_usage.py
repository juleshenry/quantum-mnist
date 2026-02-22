"""
Example Usage: Quantum Plankton Classification

This script demonstrates how to use the quantum plankton classifier.
It provides examples for both single-pair and cartesian product comparisons.

Note: Requires TensorFlow Quantum installation.
      If not installed, the script will show you the workflow without actually training.
"""

import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from plankton_quantum_algorithm import PlanktonQuantumClassifier, run_cartesian_comparison
    import numpy as np
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("\nPlease install required packages:")
    print("pip install numpy pillow")
    print("pip install tensorflow tensorflow-quantum cirq sympy")
    sys.exit(1)

def data_dir_str():
    return "data/zooplankton_0p5x"

def example_single_pair():
    """
    Example 1: Train quantum classifier on a single pair of plankton species.
    
    This is the simplest use case - binary classification between two categories.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Pair Binary Classification")
    print("="*70 + "\n")
    
    print("Configuration:")
    print("  - Image size: 8x8 (64 qubits)")
    print("  - Species: bosmina vs cyclops")
    print("  - Max images per category: 20")
    print("  - Training epochs: 10")
    print("  - Batch size: 4")
    print()
    
    # Initialize classifier
    
    classifier = PlanktonQuantumClassifier(image_size=(4, 4), seed=42) 
    
    # Define data directory (adjust path as needed)
    data_dir = data_dir_str()
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please ensure plankton dataset is available.")
        return
    
    try:
        # Train classifier
        # Modified for stability

        model, history, accuracy = classifier.train_binary_classifier(
            category_a="bosmina",
            category_b="cyclops",
            plankton_dir=data_dir,
            max_images=5, # Tiny dataset for testing
            epochs=1,     # Just one pass
            batch_size=2
        )
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Final Test Accuracy: {accuracy*100:.2f}%")
        
        # Display training history
        if hasattr(history, 'history'):
            print("\nTraining History:")
            print(f"  Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
            print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nThis might occur if:")
        print("  1. TensorFlow Quantum is not properly installed")
        print("  2. Image data is not available")
        print("  3. System resources are insufficient for quantum simulation")


def example_multiple_pairs():
    """
    Example 2: Run quantum classification on multiple species pairs.
    
    This demonstrates the "cartesian product comparison" approach,
    testing the quantum classifier across different binary classification tasks.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Pairs Comparison (Cartesian Product)")
    print("="*70 + "\n")
    
    print("Configuration:")
    print("  - Image size: 8x8 (64 qubits)")
    print("  - Number of pairs: 5")
    print("  - Max images per category: 25")
    print("  - Training epochs: 12")
    print()
    
    # Define data directory
    data_dir = data_dir_str()
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please ensure plankton dataset is available.")
        return
    
    try:
        # Run cartesian comparison
        results = run_cartesian_comparison(
            plankton_dir=data_dir,
            image_size=(8, 8),
            max_pairs=5,
            max_images=25,
            epochs=12
        )
        
        if results:
            print("\n" + "="*70)
            print("Analysis of Results")
            print("="*70)
            
            accuracies = [r['accuracy'] for r in results]
            print(f"\nMean Accuracy: {np.mean(accuracies)*100:.2f}%")
            print(f"Std Dev: {np.std(accuracies)*100:.2f}%")
            print(f"Min Accuracy: {np.min(accuracies)*100:.2f}%")
            print(f"Max Accuracy: {np.max(accuracies)*100:.2f}%")
            
            # Find best and worst pairs
            best_idx = np.argmax(accuracies)
            worst_idx = np.argmin(accuracies)
            
            print(f"\nBest performing pair:")
            print(f"  {results[best_idx]['category_a']} vs {results[best_idx]['category_b']}: "
                  f"{results[best_idx]['accuracy']*100:.2f}%")
            
            print(f"\nMost challenging pair:")
            print(f"  {results[worst_idx]['category_a']} vs {results[worst_idx]['category_b']}: "
                  f"{results[worst_idx]['accuracy']*100:.2f}%")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nThis might occur if:")
        print("  1. TensorFlow Quantum is not properly installed")
        print("  2. Image data is not available")
        print("  3. System resources are insufficient for quantum simulation")


def example_preprocessing_only():
    """
    Example 3: Demonstrate image preprocessing without training.
    
    This example shows the preprocessing pipeline and circuit encoding
    without requiring TensorFlow Quantum.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Image Preprocessing and Circuit Encoding")
    print("="*70 + "\n")
    
    # Initialize classifier
    classifier = PlanktonQuantumClassifier(image_size=(8, 8), seed=42)
    
    # Define data directory
    data_dir =  data_dir_str()
    
    if not os.path.exists(data_dir):
        print(f"Data directory ... not found: {data_dir}")
        return
    
    try:
        # Load a few images
        print("Loading sample images...")
        images = classifier.load_plankton_images(
            plankton_dir=data_dir,
            category="bosmina",
            max_images=3
        )
        
        print(f"\nLoaded {len(images)} images")
        print(f"Image shape: {images[0].shape}")
        print(f"Value range: [{images[0].min():.3f}, {images[0].max():.3f}]")
        
        # Show pixel statistics
        print("\nPixel statistics for first image:")
        print(f"  Mean: {images[0].mean():.3f}")
        print(f"  Std: {images[0].std():.3f}")
        print(f"  Pixels > 0.5 threshold: {(images[0] > 0.5).sum()}/{images[0].size}")
        
        # Try to create quantum circuit (requires Cirq)
        try:
            circuit = classifier.convert_to_circuit(images[0])
            print(f"\nQuantum circuit created successfully!")
            print(f"  Number of qubits: {len(circuit.all_qubits())}")
            print(f"  Number of operations: {len(list(circuit.all_operations()))}")
            
            # Count X gates (activated pixels)
            x_gates = sum(1 for op in circuit.all_operations() 
                         if hasattr(op.gate, '__class__') and 
                         op.gate.__class__.__name__ == 'XPowGate')
            print(f"  Number of X gates (active pixels): {x_gates}")
            
        except Exception as e:
            print(f"\nCould not create quantum circuit: {e}")
            print("Cirq may not be installed properly.")
        
    except Exception as e:
        print(f"\nError: {e}")


def main():
    """
    Main function to run examples.
    """
    print("\n" + "="*70)
    print("Quantum Plankton Classification - Example Usage")
    print("="*70)
    
    print("\nThis script demonstrates three usage examples:")
    print("  1. Single pair binary classification")
    print("  2. Multiple pairs comparison (cartesian product)")
    print("  3. Image preprocessing and circuit encoding only")
    
    print("\nNote: Examples 1 and 2 require TensorFlow Quantum installation.")
    print("      Example 3 works with just Cirq and basic dependencies.")
    
    # Check if we're in a suitable environment
    try:
        import tensorflow
        import tensorflow_quantum
        import cirq
        tfq_available = True
    except ImportError:
        tfq_available = False
        print("\n" + "!"*70)
        print("WARNING: TensorFlow Quantum not fully installed")
        print("!"*70)
        print("\nTo install required packages:")
        print("  pip install tensorflow==2.7.0 tensorflow-quantum==0.7.2 cirq sympy")
        print("\nRunning preprocessing example only...")
    
    if tfq_available:
        # Run all examples
        example_preprocessing_only()
        example_single_pair()
        example_multiple_pairs()
    else:
        # Run only preprocessing example
        example_preprocessing_only()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
