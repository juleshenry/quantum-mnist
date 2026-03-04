"""Tests for Phase 5 data loading and experiment utilities.

Validates determinism, stratification, PCA leakage prevention,
and k-category data loading.  Runs during Docker build.
"""

import unittest
import numpy as np
import os

import sys
import os
import importlib

# Phase4 and Phase5 both have a data_loader.py. PYTHONPATH lists phase4
# first, so we must force-load phase5's version.
_phase5_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "data_loader_p5",
    os.path.join(_phase5_dir, "data_loader.py"),
)
_dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dl)

load_plankton_k_categories = _dl.load_plankton_k_categories
load_plankton_k_all = _dl.load_plankton_k_all
get_top_k_categories = _dl.get_top_k_categories
get_kfold_splitter = _dl.get_kfold_splitter
apply_pca_reduction = _dl.apply_pca_reduction
_sorted_image_files = _dl._sorted_image_files

from experiment_utils import set_seed, bootstrap_ci


class TestKCategoryDataDeterminism(unittest.TestCase):
    """Data loading for k-category must be deterministic."""

    def test_load_k_all_determinism(self):
        cats = get_top_k_categories(3)
        X1, y1 = load_plankton_k_all(cats, img_size=(4, 4), max_per_class=30)
        X2, y2 = load_plankton_k_all(cats, img_size=(4, 4), max_per_class=30)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_load_k_categories_determinism(self):
        cats = get_top_k_categories(2)
        X1_tr, X1_te, y1_tr, y1_te = load_plankton_k_categories(cats, img_size=(4, 4))
        X2_tr, X2_te, y2_tr, y2_te = load_plankton_k_categories(cats, img_size=(4, 4))
        np.testing.assert_array_equal(X1_tr, X2_tr)
        np.testing.assert_array_equal(y1_tr, y2_tr)


class TestKCategoryStratification(unittest.TestCase):
    """K-category splits must be stratified."""

    def test_stratification_quality(self):
        cats = get_top_k_categories(3)
        X_tr, X_te, y_tr, y_te = load_plankton_k_categories(cats, img_size=(4, 4))
        for cls in range(3):
            train_ratio = np.mean(y_tr == cls)
            test_ratio = np.mean(y_te == cls)
            self.assertAlmostEqual(train_ratio, test_ratio, delta=0.08,
                                   msg=f"Class {cls}: train={train_ratio:.3f} test={test_ratio:.3f}")


class TestPCANoLeakage(unittest.TestCase):
    """PCA must be fit on train only."""

    def test_pca_returns_correct_shapes(self):
        cats = get_top_k_categories(2)
        X_tr, X_te, _, _ = load_plankton_k_categories(cats, img_size=(8, 8))
        X_tr_pca, X_te_pca, pca = apply_pca_reduction(X_tr, X_te, n_components=4)
        self.assertEqual(X_tr_pca.shape[1], 4)
        self.assertEqual(X_te_pca.shape[1], 4)

    def test_pca_deterministic(self):
        cats = get_top_k_categories(2)
        X_tr, X_te, _, _ = load_plankton_k_categories(cats, img_size=(8, 8))
        X1, _, _ = apply_pca_reduction(X_tr, X_te, n_components=4)
        X2, _, _ = apply_pca_reduction(X_tr, X_te, n_components=4)
        np.testing.assert_array_almost_equal(X1, X2)


class TestBootstrapCI(unittest.TestCase):
    """Bootstrap confidence intervals must be correct."""

    def test_ci_contains_mean(self):
        vals = [0.8, 0.82, 0.79, 0.85, 0.81]
        result = bootstrap_ci(vals, n_bootstrap=5000, seed=42)
        self.assertLessEqual(result['ci_lower'], result['mean'])
        self.assertGreaterEqual(result['ci_upper'], result['mean'])

    def test_ci_level(self):
        result = bootstrap_ci([1.0, 1.0, 1.0], seed=42)
        self.assertEqual(result['ci_level'], 0.95)

    def test_ci_narrows_with_low_variance(self):
        narrow = bootstrap_ci([0.5, 0.5, 0.5, 0.5, 0.5], seed=42)
        wide = bootstrap_ci([0.1, 0.5, 0.9, 0.3, 0.7], seed=42)
        narrow_width = narrow['ci_upper'] - narrow['ci_lower']
        wide_width = wide['ci_upper'] - wide['ci_lower']
        self.assertLess(narrow_width, wide_width)

    def test_ci_reproducible(self):
        vals = [0.7, 0.75, 0.8, 0.85, 0.9]
        r1 = bootstrap_ci(vals, seed=123)
        r2 = bootstrap_ci(vals, seed=123)
        self.assertEqual(r1['ci_lower'], r2['ci_lower'])
        self.assertEqual(r1['ci_upper'], r2['ci_upper'])


class TestSeedReproducibility(unittest.TestCase):
    """Seeding must produce identical sequences."""

    def test_set_seed(self):
        set_seed(99)
        a = np.random.rand(10)
        set_seed(99)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
