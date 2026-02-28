import unittest
import numpy as np
import os
from data_loader import load_plankton_binary
from run_experiments import create_fair_classical_model, create_qnn_model, convert_to_circuit

class TestScientificRigor(unittest.TestCase):
    def test_data_loader_stratification(self):
        # Smoke test for data loader
        try:
            X_train, X_test, y_train, y_test = load_plankton_binary('dinobryon', 'nauplius', img_size=(4, 4))
            self.assertEqual(len(X_train.shape), 3)
            self.assertEqual(X_train.shape[1:], (4, 4))
            # Check for stratification (rough check)
            train_ratio = np.mean(y_train)
            test_ratio = np.mean(y_test)
            self.assertAlmostEqual(train_ratio, test_ratio, delta=0.2)
        except Exception as e:
            self.fail(f"Data loader failed: {e}")

    def test_model_parameter_counts(self):
        # Fair model
        fair_model = create_fair_classical_model()
        fair_params = fair_model.count_params()
        # QNN model (harder to count params directly from Keras model before build, 
        # but we calculated it as 48 in the logic)
        self.assertGreater(fair_params, 30)
        self.assertLess(fair_params, 100)

    def test_quantum_conversion(self):
        img = np.random.rand(4, 4)
        circuit = convert_to_circuit(img)
        # len(circuit) returns moments, we want total operations
        self.assertEqual(len(list(circuit.all_operations())), 16) 

if __name__ == '__main__':
    unittest.main()
