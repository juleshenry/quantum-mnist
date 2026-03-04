"""Tests for Phase 7 expressibility and entanglement analysis.

Validates circuit construction (both small subsystem and full 17-qubit
production architecture), fidelity sampling reproducibility,
expressibility calculation, Meyer-Wallach measure, and bootstrap CIs.
Runs during Docker build.
"""

import unittest
import numpy as np
import cirq
import sys

sys.path.append('phase7')
from quantum_rigor import (
    create_pqc_subsystem,
    create_production_pqc,
    get_haar_distribution,
    sample_pqc_fidelities,
    calculate_expressibility,
    calculate_meyer_wallach,
    bootstrap_ci,
)


class TestPQCConstruction(unittest.TestCase):
    """PQC subsystem must be correctly constructed."""

    def test_qubit_count(self):
        circuit, qubits, symbols = create_pqc_subsystem(4, n_layers=1)
        self.assertEqual(len(qubits), 4)

    def test_symbol_count_per_layer(self):
        """Each layer has 2 symbols per data qubit (XX + ZZ)."""
        _, _, symbols1 = create_pqc_subsystem(4, n_layers=1)
        _, _, symbols2 = create_pqc_subsystem(4, n_layers=2)
        # 4 qubits total, 3 data qubits, 2 symbols each
        self.assertEqual(len(symbols1), 2 * 3)  # 6
        self.assertEqual(len(symbols2), 2 * 2 * 3)  # 12

    def test_increasing_layers(self):
        for layers in [1, 2, 3]:
            _, _, syms = create_pqc_subsystem(4, n_layers=layers)
            expected = 2 * layers * 3  # 2 gates * layers * data_qubits
            self.assertEqual(len(syms), expected,
                             f"Layers={layers}: expected {expected} symbols, got {len(syms)}")


class TestProductionPQC(unittest.TestCase):
    """Production PQC must mirror Phase 5 architecture exactly."""

    def test_qubit_count(self):
        """17 qubits total: 16 data + 1 readout."""
        _, qubits, _ = create_production_pqc(n_layers=1)
        self.assertEqual(len(qubits), 17)

    def test_data_qubit_grid(self):
        """Data qubits must form a 4x4 GridQubit grid."""
        _, qubits, _ = create_production_pqc(n_layers=1)
        data_qubits = qubits[:16]
        expected = cirq.GridQubit.rect(4, 4)
        self.assertEqual(data_qubits, list(expected))

    def test_readout_qubit_position(self):
        """Readout qubit must be at GridQubit(-1, -1)."""
        _, qubits, _ = create_production_pqc(n_layers=1)
        readout = qubits[-1]
        self.assertEqual(readout, cirq.GridQubit(-1, -1))

    def test_symbols_per_layer(self):
        """Each layer has 32 symbols: 16 XX + 16 ZZ."""
        _, _, symbols1 = create_production_pqc(n_layers=1)
        _, _, symbols2 = create_production_pqc(n_layers=2)
        _, _, symbols3 = create_production_pqc(n_layers=3)
        self.assertEqual(len(symbols1), 32)
        self.assertEqual(len(symbols2), 64)
        self.assertEqual(len(symbols3), 96)

    def test_entanglement_gate_count(self):
        """16 CZ gates: 15 in data chain + 1 connecting to readout."""
        circuit, _, _ = create_production_pqc(n_layers=1)
        cz_count = sum(
            1 for op in circuit.all_operations()
            if isinstance(op.gate, cirq.CZPowGate)
        )
        self.assertEqual(cz_count, 16)

    def test_symbol_names_match_phase5(self):
        """Symbol naming convention must be xx-{l}-{i} / zz-{l}-{i}."""
        _, _, symbols = create_production_pqc(n_layers=1)
        names = [str(s) for s in symbols]
        # First layer: xx-0-0, zz-0-0, xx-0-1, zz-0-1, ..., xx-0-15, zz-0-15
        for i in range(16):
            self.assertIn(f'xx-0-{i}', names)
            self.assertIn(f'zz-0-{i}', names)

    def test_no_symbol_collisions(self):
        """All symbol names must be unique."""
        _, _, symbols = create_production_pqc(n_layers=3)
        names = [str(s) for s in symbols]
        self.assertEqual(len(names), len(set(names)))


class TestHaarDistribution(unittest.TestCase):
    """Haar distribution must be a valid probability distribution."""

    def test_normalization(self):
        _, p_haar = get_haar_distribution(4, n_bins=75)
        self.assertAlmostEqual(np.sum(p_haar), 1.0, places=5)

    def test_shape(self):
        fidelities, p_haar = get_haar_distribution(4, n_bins=100)
        self.assertEqual(len(fidelities), 100)
        self.assertEqual(len(p_haar), 100)


class TestFidelitySampling(unittest.TestCase):
    """Fidelity sampling must be reproducible with seeded RNG."""

    def test_reproducibility(self):
        circuit, qubits, symbols = create_pqc_subsystem(4, n_layers=1)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        f1 = sample_pqc_fidelities(circuit, qubits, symbols, n_samples=10, rng=rng1)
        f2 = sample_pqc_fidelities(circuit, qubits, symbols, n_samples=10, rng=rng2)
        np.testing.assert_array_almost_equal(f1, f2)

    def test_fidelity_range(self):
        circuit, qubits, symbols = create_pqc_subsystem(4, n_layers=1)
        rng = np.random.RandomState(123)
        f = sample_pqc_fidelities(circuit, qubits, symbols, n_samples=50, rng=rng)
        self.assertTrue(np.all(f >= 0.0))
        self.assertTrue(np.all(f <= 1.0))


class TestExpressibility(unittest.TestCase):
    """KL divergence must be non-negative."""

    def test_kl_non_negative(self):
        circuit, qubits, symbols = create_pqc_subsystem(4, n_layers=2)
        rng = np.random.RandomState(42)
        f = sample_pqc_fidelities(circuit, qubits, symbols, n_samples=100, rng=rng)
        kl = calculate_expressibility(f, n_qubits=4)
        self.assertGreaterEqual(kl, 0.0)


class TestMeyerWallach(unittest.TestCase):
    """Meyer-Wallach measure must return valid results."""

    def test_returns_dict(self):
        circuit, qubits, symbols = create_pqc_subsystem(4, n_layers=1)
        rng = np.random.RandomState(42)
        result = calculate_meyer_wallach(circuit, qubits, symbols, n_samples=10, rng=rng)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('values', result)
        self.assertEqual(len(result['values']), 10)

    def test_reproducibility(self):
        circuit, qubits, symbols = create_pqc_subsystem(4, n_layers=1)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        r1 = calculate_meyer_wallach(circuit, qubits, symbols, n_samples=5, rng=rng1)
        r2 = calculate_meyer_wallach(circuit, qubits, symbols, n_samples=5, rng=rng2)
        self.assertAlmostEqual(r1['mean'], r2['mean'])

    def test_entanglement_range(self):
        """Q(psi) should be in [0, 1] for normalized states."""
        circuit, qubits, symbols = create_pqc_subsystem(4, n_layers=2)
        rng = np.random.RandomState(42)
        result = calculate_meyer_wallach(circuit, qubits, symbols, n_samples=20, rng=rng)
        self.assertGreaterEqual(result['mean'], 0.0)
        self.assertLessEqual(result['mean'], 1.0)


class TestBootstrapCI(unittest.TestCase):
    """Bootstrap CI must be valid."""

    def test_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        result = bootstrap_ci([0.5, 0.6, 0.7, 0.8], rng=rng)
        self.assertLessEqual(result['ci_lower'], result['mean'])
        self.assertGreaterEqual(result['ci_upper'], result['mean'])

    def test_reproducibility(self):
        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        r1 = bootstrap_ci([0.1, 0.2, 0.3], rng=rng1)
        r2 = bootstrap_ci([0.1, 0.2, 0.3], rng=rng2)
        self.assertEqual(r1['ci_lower'], r2['ci_lower'])


if __name__ == '__main__':
    unittest.main()
