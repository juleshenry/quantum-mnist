"""Phase 5 Scientific Comparison: Rigorous multi-class quantum vs. classical.

Uses stratified K-fold cross-validation, equal sample budgets, nested
hyperparameter sweep (inner CV), baselines, comprehensive metrics, and
paired statistical tests with Holm-Bonferroni correction across K values.

Instead of raw 4x4 downsampling, images are loaded at 28x28 and
PCA-reduced to 16 features.  Both QNN and Fair Classical receive the
same PCA features, isolating the quantum-vs-classical comparison from
the compression method.
"""

import os
import json
import time
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import get_top_k_categories, load_plankton_k_all, get_kfold_splitter, apply_pca_reduction
from quantum_k_classifier import create_qnn_multiclass_model, convert_to_circuit
from classical_k_classifier import create_fair_classical_k_model
from experiment_utils import (
    set_seed, majority_baseline, random_baseline, compute_metrics,
    paired_significance_test, holm_bonferroni, log_experiment_metadata,
    save_confusion_matrix,
)


# ===================================================================
# Configuration
# ===================================================================

K_VALUES = [2, 3, 4, 5]
N_FOLDS = int(os.environ.get('N_FOLDS', 5))
EPOCHS = 20
BATCH_SIZE = 32
Q_SAMPLES = int(os.environ.get('Q_SAMPLES', 400))
IS_SMOKE = os.environ.get('SMOKE_TEST', 'false').lower() == 'true'
RESULTS_DIR = os.path.join('phase5', os.environ.get('RESULTS_DIR', 'results'))

Q_SWEEP = {
    'n_layers': [1, 2],
    'learning_rate': [0.01, 0.05],
}
C_SWEEP = {
    'hidden_units': [1, 2],
    'learning_rate': [0.01, 0.05],
}

if IS_SMOKE:
    print("!!! SMOKE TEST MODE !!!")
    K_VALUES = [2]
    N_FOLDS = 2
    Q_SAMPLES = 10
    EPOCHS = 2


# ===================================================================
# Early Stopping
# ===================================================================

def _early_stopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)


# ===================================================================
# Hyperparameter Sweep (inner validation)
# ===================================================================

def run_sweep(k, model_type, x_train, y_train):
    """Run hyperparameter sweep using a stratified inner split.

    Returns best hyperparameters based on validation accuracy.
    """
    print(f"  Sweeping {model_type} for K={k}...")

    # Stratified 80/20 inner split for sweep validation
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(ss.split(np.zeros(len(y_train)), y_train))

    best_acc = -1
    best_params = None

    sweep_space = Q_SWEEP if model_type == 'quantum' else C_SWEEP
    keys, values = zip(*sweep_space.items())

    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        set_seed(42)

        if model_type == 'quantum':
            model = create_qnn_multiclass_model(k, **params)
            model.fit(x_train[tr_idx], y_train[tr_idx],
                      epochs=5, batch_size=BATCH_SIZE, verbose=0)
            acc = model.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[1]
        else:
            x_np = x_train if not isinstance(x_train, tf.Tensor) else x_train.numpy()
            model = create_fair_classical_k_model(k, **params)
            model.fit(x_np[tr_idx], y_train[tr_idx],
                      epochs=5, batch_size=BATCH_SIZE, verbose=0)
            acc = model.evaluate(x_np[val_idx], y_train[val_idx], verbose=0)[1]

        if acc > best_acc:
            best_acc = acc
            best_params = params

    print(f"    Best {model_type}: {best_params} (val_acc={best_acc:.4f})")
    return best_params


# ===================================================================
# Single Fold
# ===================================================================

def run_fold(k, X_28, y, train_idx, test_idx, fold_id, best_q_params, best_c_params):
    """Run quantum and classical models on one CV fold.

    PCA is applied per fold: fit on train, transform on test.
    Both QNN and Fair Classical receive the same 16 PCA features.
    """
    fold_seed = 42 + fold_id
    set_seed(fold_seed)

    # --- Subsample training to Q_SAMPLES for fairness ---
    train_limit = min(len(train_idx), Q_SAMPLES)
    if train_limit < len(train_idx):
        ss = StratifiedShuffleSplit(n_splits=1, train_size=train_limit, random_state=fold_seed)
        sub_idx, _ = next(ss.split(train_idx, y[train_idx]))
        train_idx_limited = train_idx[sub_idx]
    else:
        train_idx_limited = train_idx

    X_train_28 = X_28[train_idx_limited]
    X_test_28 = X_28[test_idx]
    y_train = y[train_idx_limited]
    y_test = y[test_idx]

    # --- PCA: extract 16 features from 28x28 images ---
    X_train_pca, X_test_pca, pca_obj = apply_pca_reduction(
        X_train_28, X_test_28, n_components=16)

    val_split = 0.2
    fold_res = {'k': k, 'fold': fold_id, 'n_train': len(y_train), 'n_test': len(y_test),
                'pca_variance_explained': float(pca_obj.explained_variance_ratio_.sum())}

    # ---- Quantum ----
    print(f"    Converting {len(X_train_pca)} PCA features to circuits...")
    x_train_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_train_pca])
    x_test_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_test_pca])

    q_model = create_qnn_multiclass_model(k, **best_q_params)
    log_experiment_metadata(f'QNN_k{k}', q_model, len(y_train), len(y_test))

    val_size = int(len(x_train_circ) * val_split)
    train_size = len(x_train_circ) - val_size

    start = time.time()
    q_model.fit(
        x_train_circ[:train_size], y_train[:train_size],
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        validation_data=(x_train_circ[train_size:], y_train[train_size:]),
        callbacks=[_early_stopping()],
    )
    fold_res['q_time'] = time.time() - start
    q_pred = np.argmax(q_model.predict(x_test_circ, verbose=0), axis=1)
    q_metrics = compute_metrics(y_test, q_pred, k=k)
    fold_res['q_acc'] = q_metrics['accuracy']
    fold_res['q_f1'] = q_metrics['macro_f1']
    fold_res['q_cm'] = q_metrics['confusion_matrix']

    # ---- Classical (16 PCA features) ----
    c_model = create_fair_classical_k_model(k, input_shape=(16,), **best_c_params)
    log_experiment_metadata(f'FairMLP_k{k}', c_model, len(y_train), len(y_test))

    start = time.time()
    c_model.fit(
        X_train_pca, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        validation_split=val_split, callbacks=[_early_stopping()],
    )
    fold_res['c_time'] = time.time() - start
    c_pred = np.argmax(c_model.predict(X_test_pca, verbose=0), axis=1)
    c_metrics = compute_metrics(y_test, c_pred, k=k)
    fold_res['c_acc'] = c_metrics['accuracy']
    fold_res['c_f1'] = c_metrics['macro_f1']
    fold_res['c_cm'] = c_metrics['confusion_matrix']

    # ---- Baselines ----
    fold_res['majority_baseline'] = majority_baseline(y_test)
    rb = random_baseline(y_test, k=k)
    fold_res['random_baseline'] = rb['analytical']

    print(f"  Fold {fold_id}: Q_acc={fold_res['q_acc']:.3f}  C_acc={fold_res['c_acc']:.3f}  "
          f"majority={fold_res['majority_baseline']:.3f}")

    return fold_res


# ===================================================================
# Main Comparison
# ===================================================================

def perform_comparison():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_fold_results = []
    all_summaries = []

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K={k} Scientific Comparison ({N_FOLDS}-fold CV, Q_SAMPLES={Q_SAMPLES})")
        print(f"{'='*60}")

        categories = get_top_k_categories(k)

        # Load at 28x28 for PCA and CNN
        X_28, y = load_plankton_k_all(categories, img_size=(28, 28))

        kfold = get_kfold_splitter(n_folds=N_FOLDS)

        # --- Sweep on first fold's training data ---
        first_train_idx, _ = next(iter(kfold.split(X_28, y)))
        sweep_limit = min(len(first_train_idx), Q_SAMPLES)
        if sweep_limit < len(first_train_idx):
            ss = StratifiedShuffleSplit(n_splits=1, train_size=sweep_limit, random_state=42)
            sub_idx, _ = next(ss.split(first_train_idx, y[first_train_idx]))
            sweep_train_idx = first_train_idx[sub_idx]
        else:
            sweep_train_idx = first_train_idx

        # Apply PCA for sweep data (use a held-out portion as "test" for PCA fit)
        sweep_ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        sw_tr, sw_te = next(sweep_ss.split(np.zeros(len(sweep_train_idx)), y[sweep_train_idx]))
        X_sweep_pca_tr, X_sweep_pca_te, _pca = apply_pca_reduction(
            X_28[sweep_train_idx[sw_tr]], X_28[sweep_train_idx[sw_te]], n_components=16)
        # Recombine for sweep (both portions were PCA-transformed)
        X_sweep_all_pca = np.vstack([X_sweep_pca_tr, X_sweep_pca_te])
        y_sweep = np.concatenate([y[sweep_train_idx[sw_tr]], y[sweep_train_idx[sw_te]]])

        # Quantum sweep needs circuit tensors
        x_sweep_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_sweep_all_pca])
        best_q_params = run_sweep(k, 'quantum', x_sweep_circ, y_sweep)

        best_c_params = run_sweep(k, 'classical', X_sweep_all_pca, y_sweep)

        # --- Run folds ---
        fold_results_k = []
        for fold_id, (train_idx, test_idx) in enumerate(kfold.split(X_28, y)):
            res = run_fold(k, X_28, y, train_idx, test_idx, fold_id,
                           best_q_params, best_c_params)

            # Save confusion matrices
            cm_dir = os.path.join(RESULTS_DIR, 'confusion_matrices', f'k{k}')
            for model_key in ['q', 'c']:
                cm = res.pop(f'{model_key}_cm')
                save_confusion_matrix(cm, os.path.join(cm_dir, f'{model_key}_fold{fold_id}.csv'))

            fold_results_k.append(res)

        all_fold_results.extend(fold_results_k)

        # --- Aggregate for this K ---
        df_k = pd.DataFrame(fold_results_k)
        summary = {
            'k': k,
            'categories': ','.join(categories),
            'best_q_params': str(best_q_params),
            'best_c_params': str(best_c_params),
        }
        for metric in ['q_acc', 'q_f1', 'c_acc', 'c_f1', 'q_time', 'c_time']:
            summary[f'{metric}_mean'] = df_k[metric].mean()
            summary[f'{metric}_std'] = df_k[metric].std()

        summary['majority_baseline'] = df_k['majority_baseline'].mean()
        summary['random_baseline'] = df_k['random_baseline'].mean()

        sig = paired_significance_test(df_k['q_acc'].values, df_k['c_acc'].values)
        summary['qnn_vs_fair_pvalue'] = sig['p_value']
        summary['qnn_vs_fair_test'] = sig['test_used']

        all_summaries.append(summary)

        print(f"\nK={k} Summary: Q_acc={summary['q_acc_mean']:.3f}+/-{summary['q_acc_std']:.3f}  "
              f"C_acc={summary['c_acc_mean']:.3f}+/-{summary['c_acc_std']:.3f}  "
              f"p={summary['qnn_vs_fair_pvalue']:.4f}")

    # --- Multiple comparison correction across K values ---
    if len(all_summaries) > 1:
        raw_pvals = {str(s['k']): s['qnn_vs_fair_pvalue'] for s in all_summaries}
        corrected = holm_bonferroni(raw_pvals)
        for s in all_summaries:
            key = str(s['k'])
            s['corrected_p'] = corrected[key]['corrected_p']
            s['significant_05'] = corrected[key]['significant_05']
        print(f"\nHolm-Bonferroni correction across {len(all_summaries)} K values:")
        for key, vals in corrected.items():
            print(f"  K={key}: raw_p={vals['raw_p']:.4f} -> corrected_p={vals['corrected_p']:.4f}"
                  f"  {'*' if vals['significant_05'] else 'ns'}")
    elif len(all_summaries) == 1:
        s = all_summaries[0]
        s['corrected_p'] = s['qnn_vs_fair_pvalue']
        s['significant_05'] = s['qnn_vs_fair_pvalue'] < 0.05

    # --- Save ---
    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(os.path.join(RESULTS_DIR, 'scientific_k_comparison.csv'), index=False)

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'scientific_k_summary.csv'), index=False)

    config = {
        'k_values': K_VALUES, 'n_folds': N_FOLDS, 'epochs': EPOCHS,
        'batch_size': BATCH_SIZE, 'q_samples': Q_SAMPLES,
        'q_sweep': Q_SWEEP, 'c_sweep': C_SWEEP, 'smoke_test': IS_SMOKE,
        'pca': {
            'source_resolution': '28x28',
            'n_components': 16,
            'whiten': True,
            'scaling': 'MinMaxScaler to [0, 1]',
            'note': 'PCA fit on train fold, transform on test fold. '
                    'Both QNN and Fair Classical receive the same 16 PCA features.',
        },
    }
    with open(os.path.join(RESULTS_DIR, 'scientific_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # --- Plot ---
    _plot_results(df_summary)

    print(f"\nResults saved to {RESULTS_DIR}/")


def _plot_results(df_summary):
    """Generate accuracy/F1 comparison plots with error bars and significance markers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    k_vals = df_summary['k'].values

    for ax, q_mean, q_std, c_mean, c_std, ylabel, title in [
        (ax1, 'q_acc_mean', 'q_acc_std', 'c_acc_mean', 'c_acc_std',
         'Test Accuracy', 'Accuracy Comparison'),
        (ax2, 'q_f1_mean', 'q_f1_std', 'c_f1_mean', 'c_f1_std',
         'Macro F1-Score', 'F1-Score Comparison'),
    ]:
        ax.errorbar(k_vals, df_summary[q_mean], yerr=df_summary[q_std],
                    label='Quantum', marker='o', capsize=5)
        ax.errorbar(k_vals, df_summary[c_mean], yerr=df_summary[c_std],
                    label='Classical Fair', marker='s', capsize=5)

        # Baselines
        ax.plot(k_vals, df_summary['majority_baseline'], '--', color='gray',
                label='Majority Baseline', alpha=0.7)
        ax.plot(k_vals, df_summary['random_baseline'], ':', color='gray',
                label='Random Baseline', alpha=0.7)

        # Significance markers
        if 'significant_05' in df_summary.columns:
            for i, row in df_summary.iterrows():
                if row.get('significant_05', False):
                    y_pos = max(row[q_mean], row[c_mean]) + max(row[q_std], row[c_std]) + 0.02
                    ax.annotate('*', (row['k'], y_pos), ha='center', fontsize=14, fontweight='bold')

        ax.set_xlabel('Number of Categories (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} ({N_FOLDS}-fold CV)')
        ax.set_xticks(k_vals)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'scientific_scaling_plot.png'), dpi=150)
    print(f"Plot saved to {RESULTS_DIR}/scientific_scaling_plot.png")


if __name__ == "__main__":
    perform_comparison()
