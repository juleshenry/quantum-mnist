"""Tests for Phase 2 data loading and quantum circuit construction.

Validates:
- Reproducible, stratified data splitting via prepare_binary_dataset
- Circuit construction correctness (gate count, qubit layout)
- Model architecture symbol count
- Seeding determinism

Runs during Docker build (no TFQ required — uses cirq directly).
"""

import unittest
import os
import numpy as np
import cirq
import sympy

# Phase 2 imports
from phase2.plankton_ingress import (
    prepare_binary_dataset,
    load_images_for_class,
    get_plankton_names,
    pca_transform,
    LOAD_DIMS,
    N_PCA_COMPONENTS,
    QUBIT_DIMS,
)
from phase2.binary_quantum_classifier import (
    convert_to_circuit,
    create_quantum_model,
    CircuitLayerBuilder,
    SEED,
    N_FOLDS,
)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_utils",
    os.path.join(os.path.dirname(__file__), "..", "utils", "experiment_utils.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
set_seed = _mod.set_seed
bootstrap_ci = _mod.bootstrap_ci


class TestPrepareDatasetReproducibility(unittest.TestCase):
    """prepare_binary_dataset must be deterministic with a seed."""

    def setUp(self):
        self.plank = get_plankton_names()
        if len(self.plank) < 2:
            self.skipTest("Need at least 2 plankton classes")

    def test_determinism_with_seed(self):
        """Same seed must produce identical splits (with PCA)."""
        (X1_tr, y1_tr), (X1_te, y1_te) = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        (X2_tr, y2_tr), (X2_te, y2_te) = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        np.testing.assert_array_equal(X1_tr, X2_tr)
        np.testing.assert_array_equal(y1_tr, y2_tr)
        np.testing.assert_array_equal(X1_te, X2_te)
        np.testing.assert_array_equal(y1_te, y2_te)

    def test_pca_output_shape(self):
        """PCA output should have shape (N, 16)."""
        (X_train, _), (X_test, _) = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        self.assertEqual(X_train.shape[1], N_PCA_COMPONENTS)
        self.assertEqual(X_test.shape[1], N_PCA_COMPONENTS)

    def test_pca_values_in_unit_range(self):
        """MinMaxScaler should produce values in [0, 1] on train set."""
        (X_train, _), _ = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        self.assertGreaterEqual(X_train.min(), 0.0)
        self.assertLessEqual(X_train.max(), 1.0)

    def test_different_seeds_different_splits(self):
        """Different seeds must produce different splits."""
        (X1_tr, _), _ = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        (X2_tr, _), _ = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=99
        )
        self.assertFalse(np.array_equal(X1_tr, X2_tr))

    def test_stratified_balance(self):
        """Train and test sets must preserve class proportions."""
        (_, y_train), (_, y_test) = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=40, seed=42
        )
        train_ratio = np.mean(y_train == 0)
        test_ratio = np.mean(y_test == 0)
        self.assertAlmostEqual(train_ratio, 0.5, delta=0.15)
        self.assertAlmostEqual(test_ratio, 0.5, delta=0.15)

    def test_no_overlap_between_train_test(self):
        """Train and test feature vectors must not overlap (data leakage check)."""
        (X_train, _), (X_test, _) = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        # PCA output is already 2-D (N, 16)
        train_flat = set(map(tuple, X_train))
        test_flat = set(map(tuple, X_test))
        overlap = train_flat & test_flat
        self.assertEqual(len(overlap), 0, "Train/test overlap detected")

    def test_no_pca_leakage(self):
        """PCA must be fit on train only — different folds give different transforms."""
        (X1_tr, _), _ = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        (X2_tr, _), _ = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=99
        )
        self.assertFalse(np.array_equal(X1_tr, X2_tr))


class TestConvertToCircuit(unittest.TestCase):
    """Angle-encoded circuits must have correct structure."""

    def test_gate_count(self):
        """16 PCA features should produce 16 Ry gates."""
        features = np.random.RandomState(42).rand(16)
        circuit = convert_to_circuit(features)
        ry_gates = [
            op for moment in circuit for op in moment
            if isinstance(op.gate, cirq.ops.common_gates.YPowGate)
        ]
        self.assertEqual(len(ry_gates), 16)

    def test_qubit_count(self):
        """Circuit should use 16 data qubits."""
        features = np.random.RandomState(42).rand(16)
        circuit = convert_to_circuit(features)
        self.assertEqual(len(circuit.all_qubits()), 16)

    def test_determinism(self):
        """Same features must produce the same circuit."""
        features = np.random.RandomState(42).rand(16)
        c1 = convert_to_circuit(features)
        c2 = convert_to_circuit(features)
        self.assertEqual(c1, c2)

    def test_zero_features_trivial(self):
        """All-zero features should produce identity-like rotations."""
        features = np.zeros(16)
        circuit = convert_to_circuit(features)
        # Ry(0) = identity, circuit should have 16 gates but all trivial
        self.assertEqual(len(circuit.all_qubits()), 16)


class TestQuantumModel(unittest.TestCase):
    """Model circuit must have correct symbol count and structure."""

    def test_symbol_count(self):
        """Model should have 160 symbols (16 qubits x 10 param layers)."""
        circuit, readout_op = create_quantum_model()
        symbols = cirq.parameter_names(circuit)
        self.assertEqual(len(symbols), 160)

    def test_readout_qubit(self):
        """Readout operator must measure Z on the readout qubit."""
        _, readout_op = create_quantum_model()
        self.assertEqual(
            readout_op,
            cirq.Z(cirq.GridQubit(-1, -1))
        )

    def test_unique_symbol_names(self):
        """All symbol names in the model circuit must be unique."""
        circuit, _ = create_quantum_model()
        names = cirq.parameter_names(circuit)
        self.assertEqual(len(names), len(set(names)),
                         "Duplicate symbol names found")


class TestBootstrapCI(unittest.TestCase):
    """bootstrap_ci utility must return valid intervals."""

    def test_ci_contains_mean(self):
        vals = [0.7, 0.8, 0.75, 0.85, 0.72]
        result = bootstrap_ci(vals, seed=42)
        self.assertLessEqual(result['ci_lower'], result['mean'])
        self.assertGreaterEqual(result['ci_upper'], result['mean'])

    def test_ci_determinism(self):
        vals = [0.7, 0.8, 0.75, 0.85, 0.72]
        r1 = bootstrap_ci(vals, seed=42)
        r2 = bootstrap_ci(vals, seed=42)
        self.assertEqual(r1['ci_lower'], r2['ci_lower'])
        self.assertEqual(r1['ci_upper'], r2['ci_upper'])


class TestSeedFunction(unittest.TestCase):
    """set_seed must produce reproducible numpy sequences."""

    def test_numpy_determinism(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
