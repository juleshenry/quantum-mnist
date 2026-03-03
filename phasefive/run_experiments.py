"""Phase 5: Rigorous K-category scaling experiments.

Uses stratified K-fold cross-validation, equal sample budgets across
all models, early stopping, baselines, comprehensive metrics, and
paired statistical tests with Holm-Bonferroni correction.

This is the standard multi-class experiment runner.  For the swept
hyperparameter comparison, see ``scientific_comparison.py``.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import load_plankton_k_all, get_top_k_categories, get_kfold_splitter
from quantum_k_classifier import create_qnn_multiclass_model, convert_to_circuit
from classical_k_classifier import create_fair_classical_k_model, create_cnn_k_model
from experiment_utils import (
    set_seed, majority_baseline, random_baseline, compute_metrics,
    paired_significance_test, holm_bonferroni, log_experiment_metadata,
    save_confusion_matrix,
)


# ===================================================================
# Configuration
# ===================================================================

K_VALUES = [2, 3, 4, 5, 8, 12, 16]
N_FOLDS = int(os.environ.get('N_FOLDS', 5))
EPOCHS = 20
BATCH_SIZE = 32
Q_SAMPLES = int(os.environ.get('Q_SAMPLES', 400))
IS_SMOKE = os.environ.get('SMOKE_TEST', 'false').lower() == 'true'
RESULTS_DIR = os.path.join('phasefive', os.environ.get('RESULTS_DIR', 'results'))

if IS_SMOKE:
    print("!!! SMOKE TEST MODE !!!")
    K_VALUES = [2, 3]
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
# Single Fold
# ===================================================================

def run_fold(k, X_4, X_28, y, train_idx, test_idx, fold_id):
    """Run QNN, Fair Classical, and CNN on one CV fold."""
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

    X_train_4 = X_4[train_idx_limited]
    X_test_4 = X_4[test_idx]
    X_train_28 = X_28[train_idx_limited]
    X_test_28 = X_28[test_idx]
    y_train = y[train_idx_limited]
    y_test = y[test_idx]

    val_split = 0.2
    fold_res = {'k': k, 'fold': fold_id, 'n_train': len(y_train), 'n_test': len(y_test)}

    # ---- 1. QNN (4x4) ----
    x_train_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_train_4])
    x_test_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_test_4])

    qnn = create_qnn_multiclass_model(k)
    log_experiment_metadata(f'QNN_k{k}', qnn, len(y_train), len(y_test))

    val_size = int(len(x_train_circ) * val_split)
    train_size = len(x_train_circ) - val_size

    start = time.time()
    qnn.fit(
        x_train_circ[:train_size], y_train[:train_size],
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        validation_data=(x_train_circ[train_size:], y_train[train_size:]),
        callbacks=[_early_stopping()],
    )
    fold_res['qnn_time'] = time.time() - start
    q_pred = np.argmax(qnn.predict(x_test_circ, verbose=0), axis=1)
    q_metrics = compute_metrics(y_test, q_pred, k=k)
    fold_res['qnn_acc'] = q_metrics['accuracy']
    fold_res['qnn_f1'] = q_metrics['macro_f1']
    fold_res['qnn_cm'] = q_metrics['confusion_matrix']

    # ---- 2. Fair Classical (4x4) ----
    X_tr_c = X_train_4[..., np.newaxis]
    X_te_c = X_test_4[..., np.newaxis]

    fair_nn = create_fair_classical_k_model(k)
    log_experiment_metadata(f'FairMLP_k{k}', fair_nn, len(y_train), len(y_test))

    start = time.time()
    fair_nn.fit(
        X_tr_c, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        validation_split=val_split, callbacks=[_early_stopping()],
    )
    fold_res['fair_time'] = time.time() - start
    f_pred = np.argmax(fair_nn.predict(X_te_c, verbose=0), axis=1)
    f_metrics = compute_metrics(y_test, f_pred, k=k)
    fold_res['fair_acc'] = f_metrics['accuracy']
    fold_res['fair_f1'] = f_metrics['macro_f1']
    fold_res['fair_cm'] = f_metrics['confusion_matrix']

    # ---- 3. CNN (28x28) ----
    X_tr_28 = X_train_28[..., np.newaxis]
    X_te_28 = X_test_28[..., np.newaxis]

    cnn = create_cnn_k_model(k)
    log_experiment_metadata(f'CNN_k{k}', cnn, len(y_train), len(y_test))

    start = time.time()
    cnn.fit(
        X_tr_28, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        validation_split=val_split, callbacks=[_early_stopping()],
    )
    fold_res['cnn_time'] = time.time() - start
    c_pred = np.argmax(cnn.predict(X_te_28, verbose=0), axis=1)
    c_metrics = compute_metrics(y_test, c_pred, k=k)
    fold_res['cnn_acc'] = c_metrics['accuracy']
    fold_res['cnn_f1'] = c_metrics['macro_f1']
    fold_res['cnn_cm'] = c_metrics['confusion_matrix']

    # ---- Baselines ----
    fold_res['majority_baseline'] = majority_baseline(y_test)
    rb = random_baseline(y_test, k=k)
    fold_res['random_baseline'] = rb['analytical']

    print(f"  Fold {fold_id}: QNN={fold_res['qnn_acc']:.3f}  Fair={fold_res['fair_acc']:.3f}  "
          f"CNN={fold_res['cnn_acc']:.3f}  majority={fold_res['majority_baseline']:.3f}")

    return fold_res


# ===================================================================
# Full K Experiment
# ===================================================================

def run_k_experiment(k):
    """Run K-fold CV for a given number of categories."""
    print(f"\n{'='*60}")
    print(f"K={k} Scaling Experiment ({N_FOLDS}-fold CV, Q_SAMPLES={Q_SAMPLES})")
    print(f"{'='*60}")

    categories = get_top_k_categories(k)
    X_4, y = load_plankton_k_all(categories, img_size=(4, 4))
    X_28, _ = load_plankton_k_all(categories, img_size=(28, 28))

    kfold = get_kfold_splitter(n_folds=N_FOLDS)
    fold_results = []

    for fold_id, (train_idx, test_idx) in enumerate(kfold.split(X_4, y)):
        res = run_fold(k, X_4, X_28, y, train_idx, test_idx, fold_id)

        # Save confusion matrices
        cm_dir = os.path.join(RESULTS_DIR, 'confusion_matrices', f'k{k}')
        for model_key in ['qnn', 'fair', 'cnn']:
            cm = res.pop(f'{model_key}_cm')
            save_confusion_matrix(cm, os.path.join(cm_dir, f'{model_key}_fold{fold_id}.csv'))

        fold_results.append(res)

    return fold_results


# ===================================================================
# Plotting
# ===================================================================

def plot_results(df_summary):
    """Generate scaling comparison plots with baselines and significance markers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    k_vals = df_summary['k'].values

    for ax, models, ylabel, title in [
        (ax1, [('qnn_acc', 'QNN'), ('fair_acc', 'Fair Classical'), ('cnn_acc', 'CNN')],
         'Test Accuracy', 'Accuracy vs Categories'),
        (ax2, [('qnn_f1', 'QNN'), ('fair_f1', 'Fair Classical'), ('cnn_f1', 'CNN')],
         'Macro F1-Score', 'F1-Score vs Categories'),
    ]:
        for prefix, label in models:
            ax.errorbar(k_vals, df_summary[f'{prefix}_mean'], yerr=df_summary[f'{prefix}_std'],
                        label=label, marker='o', capsize=5)

        ax.plot(k_vals, df_summary['majority_baseline'], '--', color='gray',
                label='Majority Baseline', alpha=0.7)
        ax.plot(k_vals, df_summary['random_baseline'], ':', color='gray',
                label='Random Baseline', alpha=0.7)

        if 'significant_05' in df_summary.columns:
            for _, row in df_summary.iterrows():
                if row.get('significant_05', False):
                    y_max = max(row['qnn_acc_mean'], row['fair_acc_mean'])
                    y_pos = y_max + 0.03
                    ax.annotate('*', (row['k'], y_pos), ha='center', fontsize=14, fontweight='bold')

        ax.set_xlabel('Number of Categories (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} ({N_FOLDS}-fold CV)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'k_scaling_comparison.png'), dpi=150)
    print(f"Plot saved to {RESULTS_DIR}/k_scaling_comparison.png")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print(f"Phase 5 Rigorous K-Scaling Experiments")
    print(f"  K_VALUES={K_VALUES}  N_FOLDS={N_FOLDS}  Q_SAMPLES={Q_SAMPLES}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    all_summaries = []

    for k in K_VALUES:
        try:
            fold_results = run_k_experiment(k)
            all_results.extend(fold_results)

            # Aggregate for this K
            df_k = pd.DataFrame(fold_results)
            summary = {'k': k}
            for metric in ['qnn_acc', 'qnn_f1', 'fair_acc', 'fair_f1',
                           'cnn_acc', 'cnn_f1', 'qnn_time', 'fair_time', 'cnn_time']:
                summary[f'{metric}_mean'] = df_k[metric].mean()
                summary[f'{metric}_std'] = df_k[metric].std()

            summary['majority_baseline'] = df_k['majority_baseline'].mean()
            summary['random_baseline'] = df_k['random_baseline'].mean()

            sig = paired_significance_test(df_k['qnn_acc'].values, df_k['fair_acc'].values)
            summary['qnn_vs_fair_pvalue'] = sig['p_value']
            summary['qnn_vs_fair_test'] = sig['test_used']

            all_summaries.append(summary)

        except Exception as e:
            print(f"FAILED K={k}: {e}")
            import traceback; traceback.print_exc()

    # --- Multiple-comparison correction ---
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
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_k_results.csv'), index=False)

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_k_summary.csv'), index=False)

    config = {
        'k_values': K_VALUES, 'n_folds': N_FOLDS, 'epochs': EPOCHS,
        'batch_size': BATCH_SIZE, 'q_samples': Q_SAMPLES, 'smoke_test': IS_SMOKE,
    }
    with open(os.path.join(RESULTS_DIR, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n--- Final Summary ---")
    print(df_summary[['k', 'qnn_acc_mean', 'fair_acc_mean', 'cnn_acc_mean']].to_string())

    plot_results(df_summary)

    print(f"\nAll results saved to {RESULTS_DIR}/")
