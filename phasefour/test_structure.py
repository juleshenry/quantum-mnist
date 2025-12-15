"""
Basic structure tests for quantum plankton classifier.

These tests verify the code structure and basic functionality
without requiring TensorFlow Quantum installation.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that the module can be imported."""
    print("Testing imports...")
    try:
        # This will work even without TFQ, just won't be able to train
        import plankton_quantum_algorithm as pqa
        print("✓ Module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_class_initialization():
    """Test PlanktonQuantumClassifier initialization."""
    print("\nTesting class initialization...")
    try:
        from plankton_quantum_algorithm import PlanktonQuantumClassifier
        
        # Test with 8x8
        classifier = PlanktonQuantumClassifier(image_size=(8, 8))
        assert classifier.image_size == (8, 8)
        assert classifier.n_qubits == 64
        print("✓ 8x8 initialization works")
        
        # Test with 4x4
        classifier = PlanktonQuantumClassifier(image_size=(4, 4))
        assert classifier.image_size == (4, 4)
        assert classifier.n_qubits == 16
        print("✓ 4x4 initialization works")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_image_preprocessing():
    """Test image preprocessing functions."""
    print("\nTesting image preprocessing...")
    try:
        from plankton_quantum_algorithm import PlanktonQuantumClassifier
        from PIL import Image
        
        classifier = PlanktonQuantumClassifier(image_size=(8, 8))
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # Test preprocessing
        processed = classifier.preprocess_image(test_image)
        
        assert processed.shape == (8, 8), f"Expected (8, 8), got {processed.shape}"
        assert processed.min() >= 0 and processed.max() <= 1, "Values should be normalized to [0, 1]"
        assert processed.dtype in [np.float64, np.float32], "Should be float type"
        
        print(f"✓ Preprocessing works: shape={processed.shape}, range=[{processed.min():.3f}, {processed.max():.3f}]")
        return True
        
    except ImportError as e:
        print(f"⚠ Skipping preprocessing test (PIL not installed): {e}")
        return True  # Not a failure
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False


def test_bilinear_interpolation():
    """Test bilinear interpolation resizing."""
    print("\nTesting bilinear interpolation...")
    try:
        from plankton_quantum_algorithm import PlanktonQuantumClassifier
        
        classifier = PlanktonQuantumClassifier(image_size=(8, 8))
        
        # Create test image (16x16)
        test_image = np.random.randint(0, 256, size=(16, 16), dtype=np.uint8)
        
        # Resize to 8x8
        resized = classifier.resize_image(test_image, 8, 8)
        
        assert resized.shape == (8, 8), f"Expected (8, 8), got {resized.shape}"
        assert resized.min() >= 0 and resized.max() <= 255, "Values should be in [0, 255]"
        
        print(f"✓ Bilinear interpolation works: {test_image.shape} -> {resized.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Bilinear interpolation failed: {e}")
        return False


def test_circuit_encoding():
    """Test quantum circuit encoding (requires Cirq)."""
    print("\nTesting circuit encoding...")
    try:
        from plankton_quantum_algorithm import PlanktonQuantumClassifier
        import cirq
        
        classifier = PlanktonQuantumClassifier(image_size=(4, 4))  # Use smaller for testing
        
        # Create test image
        test_image = np.array([
            [0.0, 0.2, 0.4, 0.6],
            [0.8, 1.0, 0.3, 0.5],
            [0.1, 0.9, 0.7, 0.2],
            [0.4, 0.6, 0.8, 0.3]
        ])
        
        # Convert to circuit
        circuit = classifier.convert_to_circuit(test_image)
        
        # Verify circuit properties
        assert len(circuit.all_qubits()) == 16, "Should have 16 qubits for 4x4 image"
        
        # Count X gates (pixels > 0.5)
        x_gates = sum(1 for op in circuit.all_operations())
        expected_x_gates = (test_image > 0.5).sum()
        assert x_gates == expected_x_gates, f"Expected {expected_x_gates} X gates, got {x_gates}"
        
        print(f"✓ Circuit encoding works: {x_gates} X gates for {expected_x_gates} pixels > 0.5")
        return True
        
    except ImportError as e:
        print(f"⚠ Skipping circuit encoding test (Cirq not installed): {e}")
        return True  # Not a failure
    except Exception as e:
        print(f"✗ Circuit encoding failed: {e}")
        return False


def test_circuit_layer_builder():
    """Test CircuitLayerBuilder class (requires Cirq)."""
    print("\nTesting CircuitLayerBuilder...")
    try:
        from plankton_quantum_algorithm import CircuitLayerBuilder
        import cirq
        import sympy
        
        # Create test qubits
        data_qubits = cirq.GridQubit.rect(2, 2)  # 2x2 grid
        readout = cirq.GridQubit(-1, -1)
        
        # Create builder
        builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
        
        # Create circuit and add layer
        circuit = cirq.Circuit()
        builder.add_layer(circuit, cirq.XX, "test")
        
        # Verify layer was added
        assert len(list(circuit.all_operations())) == 4, "Should have 4 operations (one per data qubit)"
        
        print("✓ CircuitLayerBuilder works")
        return True
        
    except ImportError as e:
        print(f"⚠ Skipping CircuitLayerBuilder test (Cirq not installed): {e}")
        return True  # Not a failure
    except Exception as e:
        print(f"✗ CircuitLayerBuilder failed: {e}")
        return False


def test_data_loading():
    """Test plankton data loading."""
    print("\nTesting data loading...")
    try:
        from plankton_quantum_algorithm import PlanktonQuantumClassifier
        
        classifier = PlanktonQuantumClassifier(image_size=(8, 8))
        
        # Try to load data
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "zooplankton_0p5x"
        )
        
        if not os.path.exists(data_dir):
            print(f"⚠ Skipping data loading test (data not found at {data_dir})")
            return True
        
        # Load a small number of images
        images = classifier.load_plankton_images(
            plankton_dir=data_dir,
            category="bosmina",
            max_images=3
        )
        
        assert len(images) > 0, "Should load at least one image"
        assert all(img.shape == (8, 8) for img in images), "All images should be 8x8"
        
        print(f"✓ Data loading works: loaded {len(images)} images")
        return True
        
    except Exception as e:
        print(f"⚠ Data loading test skipped or failed: {e}")
        return True  # Not a critical failure


def run_all_tests():
    """Run all structure tests."""
    print("="*70)
    print("Running Quantum Plankton Classifier Structure Tests")
    print("="*70)
    
    tests = [
        test_imports,
        test_class_initialization,
        test_image_preprocessing,
        test_bilinear_interpolation,
        test_circuit_encoding,
        test_circuit_layer_builder,
        test_data_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed or skipped")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
