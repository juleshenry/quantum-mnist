"""Tests for Phase 3 hyperparameter optimization components.

Validates:
- Multi-encoding circuit construction (angle vs basis)
- Variable-depth model architecture
- Hyperparameter sweep generation completeness
- Nested CV structural properties (no data leakage by construction)

Runs during Docker build (no TFQ required — uses cirq directly).
"""

import unittest
import os
import numpy as np
import cirq
import sympy
from sklearn.model_selection import StratifiedKFold

# Phase 3 imports — use importlib to avoid top-level side effects
import importlib.util

_phase3_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "optimize_binary_classifier",
    os.path.join(_phase3_dir, "optimize_binary_classifier.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

convert_to_circuit = _mod.convert_to_circuit
create_quantum_model = _mod.create_quantum_model
setup_sweep = _mod.setup_sweep
CircuitLayerBuilder = _mod.CircuitLayerBuilder
HYPERPARAMS = _mod.HYPERPARAMS
SEED = _mod.SEED
OUTER_FOLDS = _mod.OUTER_FOLDS
INNER_FOLDS = _mod.INNER_FOLDS

# Experiment utils
_eu_spec = importlib.util.spec_from_file_location(
    "experiment_utils",
    os.path.join(_phase3_dir, "..", "utils", "experiment_utils.py"),
)
_eu = importlib.util.module_from_spec(_eu_spec)
_eu_spec.loader.exec_module(_eu)
set_seed = _eu.set_seed


class TestAngleEncoding(unittest.TestCase):
    """Angle-encoded circuit must use Ry gates."""

    def test_angle_gate_count(self):
        image = np.random.RandomState(42).rand(4, 4)
        circuit = convert_to_circuit(image, encoding='angle')
        ry_gates = [
            op for moment in circuit for op in moment
            if isinstance(op.gate, cirq.ops.common_gates.YPowGate)
        ]
        self.assertEqual(len(ry_gates), 16)

    def test_angle_qubit_count(self):
        image = np.random.RandomState(42).rand(4, 4)
        circuit = convert_to_circuit(image, encoding='angle')
        self.assertEqual(len(circuit.all_qubits()), 16)


class TestBasisEncoding(unittest.TestCase):
    """Basis-encoded circuit must use X gates only where value > 0.5."""

    def test_basis_gate_count(self):
        rng = np.random.RandomState(42)
        image = rng.rand(4, 4)
        circuit = convert_to_circuit(image, encoding='basis')
        x_gates = [
            op for moment in circuit for op in moment
            if isinstance(op.gate, cirq.ops.common_gates.XPowGate)
        ]
        expected_x = int(np.sum(image.flatten() > 0.5))
        self.assertEqual(len(x_gates), expected_x)

    def test_all_zeros_no_gates(self):
        image = np.zeros((4, 4))
        circuit = convert_to_circuit(image, encoding='basis')
        self.assertEqual(len(list(circuit.all_operations())), 0)

    def test_all_ones_full_gates(self):
        image = np.ones((4, 4))
        circuit = convert_to_circuit(image, encoding='basis')
        x_gates = [
            op for moment in circuit for op in moment
            if isinstance(op.gate, cirq.ops.common_gates.XPowGate)
        ]
        self.assertEqual(len(x_gates), 16)


class TestVariableDepthModel(unittest.TestCase):
    """Model circuit depth and symbol count must scale with n_layers."""

    def test_single_layer_symbols(self):
        """1 layer: 16 XX + 16 ZZ = 32 symbols."""
        circuit, _ = create_quantum_model(n_layers=1)
        symbols = cirq.parameter_names(circuit)
        self.assertEqual(len(symbols), 32)

    def test_two_layer_symbols(self):
        """2 layers: 2 * (16 XX + 16 ZZ) = 64 symbols."""
        circuit, _ = create_quantum_model(n_layers=2)
        symbols = cirq.parameter_names(circuit)
        self.assertEqual(len(symbols), 64)

    def test_readout_qubit(self):
        _, readout_op = create_quantum_model(n_layers=1)
        self.assertEqual(readout_op, cirq.Z(cirq.GridQubit(-1, -1)))

    def test_symbol_names_no_collision(self):
        """Symbol names across layers must be unique."""
        circuit, _ = create_quantum_model(n_layers=3)
        names = cirq.parameter_names(circuit)
        self.assertEqual(len(names), len(set(names)),
                         "Duplicate symbol names found")


class TestHyperparamSweep(unittest.TestCase):
    """setup_sweep must generate all combinations."""

    def test_combination_count(self):
        combos = setup_sweep()
        expected = 1
        for values in HYPERPARAMS.values():
            expected *= len(values)
        self.assertEqual(len(combos), expected)

    def test_all_keys_present(self):
        combos = setup_sweep()
        for combo in combos:
            for key in HYPERPARAMS:
                self.assertIn(key, combo)

    def test_no_duplicates(self):
        combos = setup_sweep()
        combo_strs = [str(sorted(c.items())) for c in combos]
        self.assertEqual(len(combo_strs), len(set(combo_strs)))


class TestNestedCVStructure(unittest.TestCase):
    """Nested CV must ensure outer test sets never appear in inner folds."""

    def test_outer_test_disjoint_from_inner(self):
        """Outer test indices must not appear in any inner train/val set."""
        n = 100
        y = np.array([0] * 50 + [1] * 50)

        outer_cv = StratifiedKFold(
            n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED
        )

        for outer_train_idx, outer_test_idx in outer_cv.split(np.arange(n), y):
            outer_test_set = set(outer_test_idx)
            inner_cv = StratifiedKFold(
                n_splits=INNER_FOLDS, shuffle=True, random_state=SEED
            )
            y_outer_train = y[outer_train_idx]
            for inner_train_idx, inner_val_idx in inner_cv.split(
                outer_train_idx, y_outer_train
            ):
                # inner indices are relative to outer_train_idx
                actual_inner_train = set(outer_train_idx[inner_train_idx])
                actual_inner_val = set(outer_train_idx[inner_val_idx])
                self.assertEqual(
                    len(actual_inner_train & outer_test_set), 0,
                    "Inner train leaks into outer test"
                )
                self.assertEqual(
                    len(actual_inner_val & outer_test_set), 0,
                    "Inner val leaks into outer test"
                )

    def test_outer_folds_cover_all_samples(self):
        """Union of all outer test sets must equal the full dataset."""
        n = 100
        y = np.array([0] * 50 + [1] * 50)
        outer_cv = StratifiedKFold(
            n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED
        )
        all_test = set()
        for _, test_idx in outer_cv.split(np.arange(n), y):
            all_test.update(test_idx)
        self.assertEqual(all_test, set(range(n)))

    def test_seed_reproducibility(self):
        """Same seed must produce identical fold indices."""
        n = 80
        y = np.array([0] * 40 + [1] * 40)
        cv1 = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        cv2 = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        for (tr1, te1), (tr2, te2) in zip(
            cv1.split(np.arange(n), y), cv2.split(np.arange(n), y)
        ):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)


if __name__ == "__main__":
    unittest.main()
