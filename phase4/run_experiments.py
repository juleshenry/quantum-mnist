"""Phase 4: Rigorous binary quantum vs. classical comparison.

Uses stratified K-fold cross-validation, equal sample budgets across
all models, early stopping, majority/random baselines, comprehensive
metrics (accuracy, F1, precision, recall, confusion matrices), and
paired statistical tests with multiple-comparison correction.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import load_plankton_binary_all, get_kfold_splitter
from experiment_utils import (
    set_seed, majority_baseline, random_baseline, compute_metrics,
    paired_significance_test, holm_bonferroni, log_experiment_metadata,
    save_confusion_matrix,
)


# ===================================================================
# Configuration (overridable via environment variables)
# ===================================================================

N_FOLDS = int(os.environ.get('N_FOLDS', 5))
Q_SAMPLES = int(os.environ.get('Q_SAMPLES', 200))
IS_SMOKE = os.environ.get('SMOKE_TEST', 'false').lower() == 'true'
RESULTS_DIR = os.environ.get('RESULTS_DIR', 'phase4/results')

# 25 pairs selected by power analysis (see utils/power_analysis.py).
# Selection criteria:
#   - All 25 eligible biological classes represented (>= 80 images each)
#   - Greedy class coverage, then balanced-size fill (seed=42)
#   - 25 pairs gives 88% power to detect the observed aggregate effect
#     size (d=0.65) when using pairs as the unit of replication
#   - Ambiguous classes excluded: unknown, unknown_plankton, dirt, fish, filament
PLANKTON_PAIRS = [
    ('aphanizomenon', 'leptodora'),
    ('asplanchna', 'uroglena'),
    ('asterionella', 'diaphanosoma'),
    ('asterionella', 'rotifers'),
    ('asterionella', 'uroglena'),
    ('bosmina', 'brachionus'),
    ('bosmina', 'polyarthra'),
    ('brachionus', 'synchaeta'),
    ('ceratium', 'cyclops'),
    ('conochilus', 'daphnia'),
    ('conochilus', 'fragilaria'),
    ('conochilus', 'keratella_cochlearis'),
    ('conochilus', 'trichocerca'),
    ('cyclops', 'kellicottia'),
    ('daphnia', 'kellicottia'),
    ('daphnia', 'rotifers'),
    ('dinobryon', 'nauplius'),
    ('eudiaptomus', 'kellicottia'),
    ('eudiaptomus', 'uroglena'),
    ('fragilaria', 'keratella_cochlearis'),
    ('keratella_quadrata', 'paradileptus'),
    ('keratella_quadrata', 'rotifers'),
    ('keratella_quadrata', 'uroglena'),
    ('leptodora', 'paradileptus'),
    ('maybe_cyano', 'nauplius'),
]

if IS_SMOKE:
    print("!!! SMOKE TEST MODE ENABLED !!!")
    PLANKTON_PAIRS = PLANKTON_PAIRS[:1]
    N_FOLDS = 2
    Q_SAMPLES = 10


# ===================================================================
# Model Definitions (unchanged architectures)
# ===================================================================

def create_fair_classical_model(input_shape=(4, 4, 1)):
    """~55-parameter MLP matching QNN parameter budget."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
    )
    return model


def create_cnn_model(input_shape=(28, 28, 1)):
    """Standard CNN baseline for 28x28 inputs."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ===================================================================
# Quantum Model (unchanged architecture)
# ===================================================================

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def create_quantum_model():
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()

    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i + 1]))

    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    builder.add_layer(circuit, cirq.YY, "yy1")
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def convert_to_circuit(image):
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    return tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))


def create_qnn_model():
    model_circuit, model_readout = create_quantum_model()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])
    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy],
    )
    return model


# ===================================================================
# Early-Stopping Callback
# ===================================================================

def _early_stopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)


# ===================================================================
# Single Fold Execution
# ===================================================================

def run_single_fold(X_4, y_binary, X_28, train_idx, test_idx, fold_id):
    """Run all three models on one CV fold.

    All models train on the *same* sample budget (``Q_SAMPLES``).
    The CNN gets 28x28 resolution; Fair Classical and QNN get 4x4.

    Parameters
    ----------
    X_4 : ndarray (N, 4, 4)  -- 4x4 grayscale images for all samples
    y_binary : ndarray (N,)  -- binary labels {0, 1}
    X_28 : ndarray (N, 28, 28) -- 28x28 grayscale images (same samples)
    train_idx, test_idx : array of int -- fold indices
    fold_id : int

    Returns
    -------
    dict of per-model metrics for this fold
    """
    fold_seed = 42 + fold_id
    set_seed(fold_seed)

    # --- Subsample training data to Q_SAMPLES for fairness ---
    train_limit = min(len(train_idx), Q_SAMPLES)
    if train_limit < len(train_idx):
        # Stratified subsample of training indices
        ss = StratifiedShuffleSplit(n_splits=1, train_size=train_limit, random_state=fold_seed)
        sub_idx, _ = next(ss.split(train_idx, y_binary[train_idx]))
        train_idx_limited = train_idx[sub_idx]
    else:
        train_idx_limited = train_idx

    # --- Prepare data splits ---
    X_train_4 = X_4[train_idx_limited][..., np.newaxis]
    X_test_4 = X_4[test_idx][..., np.newaxis]
    y_train = y_binary[train_idx_limited]
    y_test = y_binary[test_idx]

    X_train_28 = X_28[train_idx_limited][..., np.newaxis]
    X_test_28 = X_28[test_idx][..., np.newaxis]

    # Hold out 20% of training fold for early-stopping validation
    val_split = 0.2

    fold_results = {'fold': fold_id, 'n_train': len(y_train), 'n_test': len(y_test)}

    # ---- 1. CNN (28x28) ----
    cnn = create_cnn_model()
    log_experiment_metadata('CNN_28x28', cnn, len(y_train), len(y_test))
    start = time.time()
    cnn.fit(X_train_28, y_train, epochs=20, batch_size=32, verbose=0,
            validation_split=val_split, callbacks=[_early_stopping()])
    fold_results['cnn_time'] = time.time() - start
    cnn_pred = (cnn.predict(X_test_28, verbose=0) > 0.5).astype(int).flatten()
    cnn_metrics = compute_metrics(y_test.astype(int), cnn_pred, k=2)
    fold_results['cnn_acc'] = cnn_metrics['accuracy']
    fold_results['cnn_f1'] = cnn_metrics['macro_f1']
    fold_results['cnn_cm'] = cnn_metrics['confusion_matrix']

    # ---- 2. Fair Classical (4x4) ----
    fair_nn = create_fair_classical_model()
    log_experiment_metadata('FairMLP_4x4', fair_nn, len(y_train), len(y_test))
    start = time.time()
    fair_nn.fit(X_train_4, y_train, epochs=20, batch_size=32, verbose=0,
                validation_split=val_split, callbacks=[_early_stopping()])
    fold_results['fair_time'] = time.time() - start
    fair_pred = (fair_nn.predict(X_test_4, verbose=0) > 0.0).astype(int).flatten()
    fair_metrics = compute_metrics(y_test.astype(int), fair_pred, k=2)
    fold_results['fair_acc'] = fair_metrics['accuracy']
    fold_results['fair_f1'] = fair_metrics['macro_f1']
    fold_results['fair_cm'] = fair_metrics['confusion_matrix']

    # ---- 3. QNN (4x4) ----
    X_train_4_flat = X_4[train_idx_limited]
    X_test_4_flat = X_4[test_idx]
    x_train_circ = [convert_to_circuit(x) for x in X_train_4_flat]
    x_test_circ = [convert_to_circuit(x) for x in X_test_4_flat]
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    qnn = create_qnn_model()
    log_experiment_metadata('QNN_4x4', qnn, len(y_train), len(y_test))
    start = time.time()
    # QNN validation split via manual index to avoid tfq tensor issues
    val_size = int(len(x_train_tfcirc) * val_split)
    train_size = len(x_train_tfcirc) - val_size
    qnn.fit(
        x_train_tfcirc[:train_size], y_train_hinge[:train_size],
        epochs=20, batch_size=32, verbose=0,
        validation_data=(x_train_tfcirc[train_size:], y_train_hinge[train_size:]),
        callbacks=[_early_stopping()],
    )
    fold_results['qnn_time'] = time.time() - start
    qnn_raw = qnn.predict(x_test_tfcirc, verbose=0).flatten()
    qnn_pred = (qnn_raw > 0.0).astype(int)
    qnn_metrics = compute_metrics(y_test.astype(int), qnn_pred, k=2)
    fold_results['qnn_acc'] = qnn_metrics['accuracy']
    fold_results['qnn_f1'] = qnn_metrics['macro_f1']
    fold_results['qnn_cm'] = qnn_metrics['confusion_matrix']

    # ---- Baselines ----
    fold_results['majority_baseline'] = majority_baseline(y_test.astype(int))
    rb = random_baseline(y_test.astype(int), k=2)
    fold_results['random_baseline'] = rb['analytical']

    print(f"  Fold {fold_id}: CNN={fold_results['cnn_acc']:.3f}  "
          f"Fair={fold_results['fair_acc']:.3f}  QNN={fold_results['qnn_acc']:.3f}  "
          f"majority={fold_results['majority_baseline']:.3f}")

    return fold_results


# ===================================================================
# Full Experiment for One Pair
# ===================================================================

def run_experiment(class_a, class_b):
    """Run K-fold CV experiment for a single plankton pair.

    Returns a summary dict with aggregated metrics, p-values, and
    a list of per-fold results.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {class_a} vs {class_b}  ({N_FOLDS}-fold CV, Q_SAMPLES={Q_SAMPLES})")
    print(f"{'='*60}")

    # Load all data at both resolutions (same images, same order)
    X_4, y = load_plankton_binary_all(class_a, class_b, img_size=(4, 4))
    X_28, _ = load_plankton_binary_all(class_a, class_b, img_size=(28, 28))

    kfold = get_kfold_splitter(n_folds=N_FOLDS)
    fold_results = []

    for fold_id, (train_idx, test_idx) in enumerate(kfold.split(X_4, y)):
        res = run_single_fold(X_4, y, X_28, train_idx, test_idx, fold_id)
        res['class_a'] = class_a
        res['class_b'] = class_b

        # Save confusion matrices
        cm_dir = os.path.join(RESULTS_DIR, 'confusion_matrices',
                              f'{class_a}_vs_{class_b}')
        for model_key in ['cnn', 'fair', 'qnn']:
            cm = res.pop(f'{model_key}_cm')
            save_confusion_matrix(cm, os.path.join(cm_dir, f'{model_key}_fold{fold_id}.csv'))

        fold_results.append(res)

    # --- Aggregate ---
    df_folds = pd.DataFrame(fold_results)
    summary = {'pair': f"{class_a}_vs_{class_b}"}

    for metric in ['cnn_acc', 'cnn_f1', 'fair_acc', 'fair_f1', 'qnn_acc', 'qnn_f1',
                   'cnn_time', 'fair_time', 'qnn_time']:
        summary[f'{metric}_mean'] = df_folds[metric].mean()
        summary[f'{metric}_std'] = df_folds[metric].std()

    summary['majority_baseline'] = df_folds['majority_baseline'].mean()
    summary['random_baseline'] = df_folds['random_baseline'].mean()

    # --- Per-pair statistical test: QNN vs Fair Classical (paired by fold) ---
    # NOTE: At n=5 folds, Wilcoxon signed-rank cannot reach p<0.05
    # (min p = 2/32 = 0.0625). The code falls back to paired t-test,
    # which requires Cohen's d >= 1.62 for 80% power. Per-pair p-values
    # are reported for transparency but the aggregate test across all
    # pairs is the primary analysis (see main block).
    sig = paired_significance_test(df_folds['qnn_acc'].values, df_folds['fair_acc'].values)
    summary['qnn_vs_fair_pvalue'] = sig['p_value']
    summary['qnn_vs_fair_test'] = sig['test_used']

    print(f"\nSummary: QNN={summary['qnn_acc_mean']:.3f}+/-{summary['qnn_acc_std']:.3f}  "
          f"Fair={summary['fair_acc_mean']:.3f}+/-{summary['fair_acc_std']:.3f}  "
          f"p={summary['qnn_vs_fair_pvalue']:.4f} ({sig['test_used']})")

    return summary, fold_results


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print(f"Phase 4 Rigorous Experiments")
    print(f"  N_FOLDS={N_FOLDS}  Q_SAMPLES={Q_SAMPLES}  SMOKE={IS_SMOKE}")
    print(f"  Results -> {RESULTS_DIR}/")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summaries = []
    all_fold_results = []

    for class_a, class_b in PLANKTON_PAIRS:
        try:
            summary, fold_res = run_experiment(class_a, class_b)
            all_summaries.append(summary)
            all_fold_results.extend(fold_res)
        except Exception as e:
            print(f"FAILED: {class_a} vs {class_b}: {e}")
            import traceback; traceback.print_exc()

    # --- Multiple-comparison correction across pairs ---
    if len(all_summaries) > 1:
        raw_pvals = {s['pair']: s['qnn_vs_fair_pvalue'] for s in all_summaries}
        corrected = holm_bonferroni(raw_pvals)
        for s in all_summaries:
            pair = s['pair']
            s['qnn_vs_fair_corrected_p'] = corrected[pair]['corrected_p']
            s['significant_05'] = corrected[pair]['significant_05']
        print(f"\nHolm-Bonferroni correction applied across {len(all_summaries)} pairs:")
        for pair, vals in corrected.items():
            print(f"  {pair}: raw_p={vals['raw_p']:.4f} -> corrected_p={vals['corrected_p']:.4f}"
                  f"  {'*' if vals['significant_05'] else 'ns'}")
    elif len(all_summaries) == 1:
        s = all_summaries[0]
        s['qnn_vs_fair_corrected_p'] = s['qnn_vs_fair_pvalue']
        s['significant_05'] = s['qnn_vs_fair_pvalue'] < 0.05

    # --- Aggregate test: pairs as unit of replication ---
    # Per-pair Wilcoxon at n=5 folds cannot reach p<0.05 (min p=0.0625).
    # Instead, we test the aggregate: is QNN systematically better/worse
    # than Fair Classical across the population of binary tasks?
    if len(all_summaries) >= 2:
        deltas = np.array([s['qnn_acc_mean'] - s['fair_acc_mean'] for s in all_summaries])
        from scipy.stats import ttest_1samp, wilcoxon as wilcoxon_1
        # One-sample t-test on pair-level mean differences
        t_stat, t_p = ttest_1samp(deltas, 0.0)
        aggregate_result = {
            'n_pairs': len(deltas),
            'mean_delta': float(np.mean(deltas)),
            'std_delta': float(np.std(deltas, ddof=1)),
            'effect_size_d': float(np.mean(deltas) / np.std(deltas, ddof=1)) if np.std(deltas, ddof=1) > 0 else 0.0,
            'ttest_statistic': float(t_stat),
            'ttest_p': float(t_p),
        }
        # Also Wilcoxon signed-rank on deltas if n >= 6
        if len(deltas) >= 6:
            try:
                w_stat, w_p = wilcoxon_1(deltas, alternative='two-sided')
                aggregate_result['wilcoxon_statistic'] = float(w_stat)
                aggregate_result['wilcoxon_p'] = float(w_p)
            except ValueError:
                aggregate_result['wilcoxon_p'] = 1.0

        # Win/loss/tie counts
        aggregate_result['qnn_wins'] = int(np.sum(deltas > 0))
        aggregate_result['qnn_losses'] = int(np.sum(deltas < 0))
        aggregate_result['ties'] = int(np.sum(deltas == 0))

        print(f"\n{'='*60}")
        print(f"AGGREGATE TEST: QNN vs Fair Classical across {len(deltas)} pairs")
        print(f"{'='*60}")
        print(f"  Mean delta (QNN - Fair): {aggregate_result['mean_delta']:+.4f}")
        print(f"  Std delta:               {aggregate_result['std_delta']:.4f}")
        print(f"  Effect size (Cohen's d): {aggregate_result['effect_size_d']:.2f}")
        print(f"  One-sample t-test:       t={t_stat:.3f}, p={t_p:.4f}")
        if 'wilcoxon_p' in aggregate_result:
            print(f"  Wilcoxon signed-rank:    p={aggregate_result['wilcoxon_p']:.4f}")
        print(f"  QNN wins: {aggregate_result['qnn_wins']}, "
              f"losses: {aggregate_result['qnn_losses']}, "
              f"ties: {aggregate_result['ties']}")

    # --- Save results ---
    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(os.path.join(RESULTS_DIR, 'experiment_results.csv'), index=False)

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'experiment_summary.csv'), index=False)

    # Save aggregate result
    if len(all_summaries) >= 2:
        with open(os.path.join(RESULTS_DIR, 'aggregate_test.json'), 'w') as f:
            json.dump(aggregate_result, f, indent=2)

    # Save config for reproducibility
    config = {
        'n_folds': N_FOLDS, 'q_samples': Q_SAMPLES, 'smoke_test': IS_SMOKE,
        'pairs': [list(p) for p in PLANKTON_PAIRS],
        'n_pairs': len(PLANKTON_PAIRS),
        'power_analysis': {
            'aggregate_effect_size_pilot': 0.65,
            'aggregate_power_at_25_pairs': 0.88,
            'min_pairs_for_80pct_power': 21,
            'per_pair_wilcoxon_min_p_at_5folds': 0.0625,
            'note': 'Per-pair tests use paired t-test fallback (n<6). '
                    'Aggregate test across pairs is the primary analysis.',
        },
    }
    with open(os.path.join(RESULTS_DIR, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nAll experiments completed.")
    print(f"  Per-fold results:   {RESULTS_DIR}/experiment_results.csv")
    print(f"  Summary:            {RESULTS_DIR}/experiment_summary.csv")
    print(f"  Aggregate test:     {RESULTS_DIR}/aggregate_test.json")
    print(f"  Config:             {RESULTS_DIR}/experiment_config.json")
    print(f"  Confusion matrices: {RESULTS_DIR}/confusion_matrices/")
