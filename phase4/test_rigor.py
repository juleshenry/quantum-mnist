"""Comprehensive scientific rigor tests for Phase 4.

These tests verify data loading determinism, stratification quality,
normalization, model parameter counts, quantum circuit correctness,
and shared experiment utilities.  They run during the Docker build
to catch problems before experiments begin.
"""

import unittest
import numpy as np
import os

from data_loader import (
    load_plankton_binary, load_plankton_binary_all,
    get_kfold_splitter, _sorted_image_files,
)
from run_experiments import (
    create_fair_classical_model, create_qnn_model, convert_to_circuit,
)
from experiment_utils import (
    majority_baseline, random_baseline, compute_metrics,
    paired_significance_test, holm_bonferroni, set_seed,
)


# Use the two largest classes for reliable testing
CLASS_A = 'dinobryon'
CLASS_B = 'nauplius'


class TestDataLoaderDeterminism(unittest.TestCase):
    """Data loading must be deterministic across calls."""

    def test_sorted_file_listing(self):
        """File listings must be sorted alphabetically."""
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
        path = os.path.join(data_dir, CLASS_A, 'training_data')
        files = _sorted_image_files(path)
        self.assertEqual(files, sorted(files),
                         "File listing is not sorted -- will cause non-deterministic splits")

    def test_load_determinism(self):
        """Two identical calls must produce identical arrays."""
        X1, _, y1, _ = load_plankton_binary(CLASS_A, CLASS_B, img_size=(4, 4))
        X2, _, y2, _ = load_plankton_binary(CLASS_A, CLASS_B, img_size=(4, 4))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_load_all_determinism(self):
        """load_plankton_binary_all is deterministic."""
        X1, y1 = load_plankton_binary_all(CLASS_A, CLASS_B, img_size=(4, 4))
        X2, y2 = load_plankton_binary_all(CLASS_A, CLASS_B, img_size=(4, 4))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestDataLoaderStratification(unittest.TestCase):
    """Train/test splits must be properly stratified."""

    def test_stratification_quality(self):
        """Class ratios in train and test must be within delta=0.05."""
        X_train, X_test, y_train, y_test = load_plankton_binary(
            CLASS_A, CLASS_B, img_size=(4, 4))
        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        self.assertAlmostEqual(train_ratio, test_ratio, delta=0.05,
                               msg=f"Stratification failed: train={train_ratio:.3f} test={test_ratio:.3f}")

    def test_kfold_stratification(self):
        """Every fold of StratifiedKFold must be well-stratified."""
        X, y = load_plankton_binary_all(CLASS_A, CLASS_B, img_size=(4, 4))
        kfold = get_kfold_splitter(n_folds=5)
        overall_ratio = np.mean(y)
        for fold_id, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            fold_ratio = np.mean(y[test_idx])
            self.assertAlmostEqual(
                fold_ratio, overall_ratio, delta=0.05,
                msg=f"Fold {fold_id} test ratio {fold_ratio:.3f} far from overall {overall_ratio:.3f}")


class TestDataNormalization(unittest.TestCase):
    """Pixel values must be normalized to [0, 1]."""

    def test_values_in_unit_range(self):
        X_train, X_test, _, _ = load_plankton_binary(CLASS_A, CLASS_B, img_size=(4, 4))
        self.assertGreaterEqual(X_train.min(), 0.0)
        self.assertLessEqual(X_train.max(), 1.0)
        self.assertGreaterEqual(X_test.min(), 0.0)
        self.assertLessEqual(X_test.max(), 1.0)


class TestDataSampling(unittest.TestCase):
    """max_per_class must correctly limit samples."""

    def test_max_per_class(self):
        X, y = load_plankton_binary_all(CLASS_A, CLASS_B, img_size=(4, 4), max_per_class=50)
        # Each class should have at most 50 samples
        for label in [0, 1]:
            count = np.sum(y == label)
            self.assertLessEqual(count, 50,
                                 msg=f"Class {label} has {count} samples, expected <= 50")


class TestModelParameterCounts(unittest.TestCase):
    """Parameter counts must match documented values."""

    def test_fair_classical_params(self):
        """Fair classical model: Flatten(16) -> Dense(3) -> Dense(1) = 16*3+3+3*1+1 = 55."""
        model = create_fair_classical_model()
        self.assertEqual(model.count_params(), 55,
                         f"Fair classical has {model.count_params()} params, expected 55")

    def test_qnn_params(self):
        """QNN has 48 trainable parameters (3 gate types x 16 qubits)."""
        model = create_qnn_model()
        # Build model with a dummy input to initialize weights
        import tensorflow_quantum as tfq
        import cirq
        dummy = tfq.convert_to_tensor([convert_to_circuit(np.zeros((4, 4)))])
        model.predict(dummy, verbose=0)
        self.assertEqual(model.count_params(), 48,
                         f"QNN has {model.count_params()} params, expected 48")


class TestQuantumCircuit(unittest.TestCase):
    """Quantum circuit encoding must be correct."""

    def test_operation_count(self):
        """4x4 image -> 16 Ry gates."""
        img = np.random.rand(4, 4)
        circuit = convert_to_circuit(img)
        ops = list(circuit.all_operations())
        self.assertEqual(len(ops), 16,
                         f"Circuit has {len(ops)} ops, expected 16")

    def test_circuit_uses_all_qubits(self):
        """All 16 data qubits should be used."""
        img = np.random.rand(4, 4)
        circuit = convert_to_circuit(img)
        qubits_used = set()
        for op in circuit.all_operations():
            for q in op.qubits:
                qubits_used.add(q)
        self.assertEqual(len(qubits_used), 16,
                         f"Circuit uses {len(qubits_used)} qubits, expected 16")

    def test_zero_image_gives_identity_rotations(self):
        """Ry(pi * 0) = Ry(0) should be identity -- circuit output should be trivial."""
        import cirq
        img = np.zeros((4, 4))
        circuit = convert_to_circuit(img)
        # Verify each gate has exponent 0 (i.e., identity)
        for op in circuit.all_operations():
            # Ry(0) is a phased rotation with exponent 0
            if hasattr(op.gate, 'exponent'):
                self.assertAlmostEqual(float(op.gate.exponent), 0.0, places=10)


class TestExperimentUtils(unittest.TestCase):
    """Shared utilities must return correct values."""

    def test_majority_baseline(self):
        y = np.array([0, 0, 0, 1, 1])
        self.assertAlmostEqual(majority_baseline(y), 0.6)

    def test_random_baseline(self):
        y = np.array([0, 0, 1, 1])
        rb = random_baseline(y, k=2)
        self.assertAlmostEqual(rb['analytical'], 0.5)

    def test_compute_metrics_shape(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        metrics = compute_metrics(y_true, y_pred, k=2)
        self.assertIn('accuracy', metrics)
        self.assertIn('macro_f1', metrics)
        self.assertEqual(metrics['confusion_matrix'].shape, (2, 2))
        self.assertEqual(len(metrics['per_class_f1']), 2)

    def test_compute_metrics_perfect(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        metrics = compute_metrics(y, y, k=3)
        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['macro_f1'], 1.0)

    def test_holm_bonferroni(self):
        """Correction should make p-values larger (more conservative)."""
        pvals = {'a': 0.01, 'b': 0.04, 'c': 0.06}
        corrected = holm_bonferroni(pvals)
        for key in pvals:
            self.assertGreaterEqual(corrected[key]['corrected_p'], pvals[key])
        # 0.06 * 1 = 0.06 (not significant)
        self.assertFalse(corrected['c']['significant_05'])

    def test_paired_test_identical(self):
        """Identical scores should yield p=1.0."""
        scores = [0.8, 0.8, 0.8, 0.8, 0.8]
        result = paired_significance_test(scores, scores)
        self.assertAlmostEqual(result['p_value'], 1.0)

    def test_set_seed_reproducibility(self):
        """Seeding should produce identical random sequences."""
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
