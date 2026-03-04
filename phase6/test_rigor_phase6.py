"""Tests for Phase 6 quantum saliency.

Validates circuit construction, symbol ordering, saliency model
construction, and seed reproducibility.  Runs during Docker build.
"""

import unittest
import numpy as np
import os
import sys
import importlib.util

sys.path.append('utils')

# Load quantum_saliency via importlib to avoid the data_loader PYTHONPATH collision
_p6_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "quantum_saliency",
    os.path.join(_p6_dir, "quantum_saliency.py"),
)
_qs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qs)
create_saliency_circuit = _qs.create_saliency_circuit

from experiment_utils import set_seed


class TestSaliencyCircuit(unittest.TestCase):
    """Saliency circuit must be correctly constructed."""

    def test_symbol_count(self):
        """Circuit must have n_features feat symbols + 2*n_layers*16 weight symbols."""
        n_features = 16
        k = 2
        n_layers = 1
        circuit, f_syms, w_syms, obs = create_saliency_circuit(n_features, k, n_layers)

        self.assertEqual(len(f_syms), 16)
        self.assertEqual(len(w_syms), 2 * n_layers * 16)  # XX + ZZ per qubit per layer
        self.assertEqual(len(obs), k)

    def test_symbol_ordering(self):
        """Feature symbols must sort before weight symbols (ControlledPQC requirement)."""
        _, f_syms, w_syms, _ = create_saliency_circuit(16, 2, 1)
        all_names = [str(s) for s in f_syms] + [str(s) for s in w_syms]
        self.assertEqual(all_names, sorted(all_names),
                         "Symbols must be in sorted order for ControlledPQC")

    def test_two_layer_circuit(self):
        """Multi-layer circuit must have more weight symbols."""
        _, _, w1, _ = create_saliency_circuit(16, 2, n_layers=1)
        _, _, w2, _ = create_saliency_circuit(16, 2, n_layers=2)
        self.assertEqual(len(w2), 2 * len(w1))


class TestSeedReproducibility(unittest.TestCase):
    """Phase 6 must be reproducible with seeding."""

    def test_seed_produces_identical_rng(self):
        set_seed(42)
        a = np.random.rand(20)
        set_seed(42)
        b = np.random.rand(20)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
