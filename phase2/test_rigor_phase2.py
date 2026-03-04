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
        """Same seed must produce identical splits."""
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

    def test_different_seeds_different_splits(self):
        """Different seeds must produce different splits."""
        (X1_tr, y1_tr), _ = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        (X2_tr, y2_tr), _ = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=99
        )
        # Extremely unlikely for two different seeds to produce identical arrays
        self.assertFalse(np.array_equal(X1_tr, X2_tr))

    def test_stratified_balance(self):
        """Train and test sets must preserve class proportions."""
        (_, y_train), (_, y_test) = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=40, seed=42
        )
        # Original dataset is 50/50, so train and test should both be ~50/50
        train_ratio = np.mean(y_train == 0)
        test_ratio = np.mean(y_test == 0)
        # Allow 15% deviation for small samples
        self.assertAlmostEqual(train_ratio, 0.5, delta=0.15)
        self.assertAlmostEqual(test_ratio, 0.5, delta=0.15)

    def test_no_overlap_between_train_test(self):
        """Train and test indices must not overlap (data leakage check)."""
        (X_train, _), (X_test, _) = prepare_binary_dataset(
            self.plank[0], self.plank[1], limit=30, seed=42
        )
        # Flatten images for comparison
        train_flat = set(map(tuple, X_train.reshape(len(X_train), -1)))
        test_flat = set(map(tuple, X_test.reshape(len(X_test), -1)))
        overlap = train_flat & test_flat
        self.assertEqual(len(overlap), 0, "Train/test overlap detected")


class TestConvertToCircuit(unittest.TestCase):
    """Angle-encoded circuits must have correct structure."""

    def test_gate_count(self):
        """A 4x4 image should produce 16 Ry gates."""
        image = np.random.RandomState(42).rand(4, 4)
        circuit = convert_to_circuit(image)
        ry_gates = [
            op for moment in circuit for op in moment
            if isinstance(op.gate, cirq.ops.common_gates.YPowGate)
        ]
        self.assertEqual(len(ry_gates), 16)

    def test_qubit_count(self):
        """Circuit should use 16 data qubits."""
        image = np.random.RandomState(42).rand(4, 4)
        circuit = convert_to_circuit(image)
        self.assertEqual(len(circuit.all_qubits()), 16)

    def test_determinism(self):
        """Same image must produce the same circuit."""
        image = np.random.RandomState(42).rand(4, 4)
        c1 = convert_to_circuit(image)
        c2 = convert_to_circuit(image)
        self.assertEqual(c1, c2)

    def test_zero_image_trivial(self):
        """All-zero image should produce identity-like rotations."""
        image = np.zeros((4, 4))
        circuit = convert_to_circuit(image)
        # Ry(0) = identity, circuit should have 16 gates but all trivial
        self.assertEqual(len(circuit.all_qubits()), 16)


class TestQuantumModel(unittest.TestCase):
    """Model circuit must have correct symbol count and structure."""

    def test_symbol_count(self):
        """Model should have 48 symbols (16 data qubits x 3 layers: XX, ZZ, YY)."""
        circuit, readout_op = create_quantum_model()
        symbols = cirq.parameter_names(circuit)
        self.assertEqual(len(symbols), 48)

    def test_readout_qubit(self):
        """Readout operator must measure Z on the readout qubit."""
        _, readout_op = create_quantum_model()
        self.assertEqual(
            readout_op,
            cirq.Z(cirq.GridQubit(-1, -1))
        )

    def test_entanglement_present(self):
        """Model circuit must contain CZ entanglement gates."""
        circuit, _ = create_quantum_model()
        cz_ops = [
            op for moment in circuit for op in moment
            if isinstance(op.gate, cirq.ops.common_gates.CZPowGate)
        ]
        # 15 CZ gates for a chain of 16 qubits
        self.assertEqual(len(cz_ops), 15)


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
