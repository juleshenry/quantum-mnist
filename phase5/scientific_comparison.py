"""
                                ★■╬▂▂▂▂▂◓□                                                           
                               ☆◕◓◊◊▇▅◕⬤▽■●⬤                                                         
                                    ▽■◑▅▆◑★■╬◒.                                                      
                                        ⬤▂▄◔▽▽▅◒◕★                                                   
                                          ⬤▄▄○◈◔◓◊○⬤▼                                                
                                           .▲◈█▆▅▇██▇▂■                                              
                                              ☆☆■◑█▇███○                                             
                                          ★★★★.  □█▲○██▄                                             
                                         ◒▇╬◑●◊■□▂▲★███◐                                             
                                         ■○╬╬■▇▄□▂ ☆◊██△                                             
                                           ★▆█▅█○▂ ▼◈█▇◑                                             
                                            ▲█★██▂  ☆██▆                                             
                                            ▼█◐◐█▂  ▲██▆                                             
                                            ★█□☆▼▽  ▲██▄                                             
                                            ★█◓     ▲██▄                                             
                                            ▲█◒     ▽◈█▆◕★                                           
                                           ◓▄○.      ▽▅██▲                                           
                                          ☆◒▅   ▽□□□■.□▇█▇△.                                         
                                          ◊○⬤   ⬤▅▄▄◊  □███▂△                                        
                                          ◊◊▄           ◑███▄△                                       
                                         ◓▇◒△           △▇███◒                                       
                                         ◐█◒            ▽△◒██▂                                       
                                         ◕█◒    ☆○◈◈◈     ◒██▂                                       
                                         ◕█◒    ▲████★    ◒██▂                                       
                                        △●○▼    ▲████★  . ◒██▅△                                      
                                      ★□▂▅◕     ▲████★  . ◒███▆△▽                                    
                                     ◓◈◊◕◑     ☆▲████▅◈   ◓█████▂◑★                                  
                                    ◕◊▼  ◒      ▲█████▂   ◓▆██████◐☆                                 
                                   ●▆▽△▄▇●      ▲████▇╬   □◊███████▂                                 
                                   ▽█.◑██●     .▼███◈◑◒   △⬤███████⬤                                 
                                    ▆☆◑██●      .▽▽▽       ■███▄███▼                                 
                                    ◔●◐▲▄●          .      ■██▇▼╬▄◐★                                 
                                       ▆█●                 ▲╬██▂☆.                                   
                                       ▆█●              .  ▼▇██▄                                     
                                   .☆□●▇●◕                 ▽▆██▇◊■☆.                                 
                                  ▽◑○⬤▼╬○△                 .□████▆█◑▲.                               
                                 ⬤◈★▽◔▅█╬▼                   ████▇███◐☆                              
                               .⬤●▽△○█▇██○▲                 ■████▂████◒▽                             
                               □▆.▼▂▅⬤╬██◕    ☆☆★★★★★☆★★.  .▂████◒◓████●                             
                              □◐▽☆◈○▲▂██◑●◓▼▲◓██████████╬◒◊███████◈□███▇◔                            
                              ⬤●□█◊▼☆◐▆█◐◓██████████████████████▆●△☆◔███◒                            
                              ⬤▆▄█⬤   ☆▂████◕◓◈◈◈◈◈◓■◈◈◈◈◒○███▆⬤▽    ▅██◒                            
                              ▽◒◊◒★    .◔▅█▇▲             ◓█▇●▲      ⬤╬◐▼                            
                                         ☆▲★               ▼▼                                        
                                                    /               
                                ___       ___  ___ (___       _ _   
                                |   )|   )|   )|   )|    |   )| | )  
                                |__/||__/ |__/||  / |__  |__/ |  /   
                                    |                                
                                                                    
                                    /           /    /             
                                ___ (  ___  ___ (    (___  ___  ___ 
                                |   )| |   )|   )|___)|    |   )|   )
                                |__/ | |__/||  / | \  |__  |__/ |  / 
                                |                                    
                                                                    
                                                /    /               
                                _ _  ___  ___ (___    ___  ___      
                                | | )|   )|    |   )| |   )|___)     
                                |  / |__/||__  |  / | |  / |__       
                                                                    
                                                                    
                                /                     /             
                                (  ___  ___  ___  ___    ___  ___    
                                | |___)|   )|   )|   )| |   )|   )   
                                | |__  |__/||    |  / | |  / |__/    
                                                            __/                                                                                                         
                                                                                                    
                                            by Julian Henry                                                        
"""

"""Phase 5 Scientific Comparison: Rigorous multi-class quantum vs. classical.

Implementes NESTED CROSS-VALIDATION for truly rigorous model assessment.
Outer Loop: 5-fold CV to estimate generalization error.
Inner Loop: 3-fold CV on each outer train set to select optimal hyperparameters.

Experimental setup:
- Fixed sample budget (Q_SAMPLES) across all models.
- PCA-based feature extraction (16 components) consistent for both.
- Statistical significance tests with Holm-Bonferroni correction.
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from data_loader import get_top_k_categories, load_plankton_k_all, apply_pca_reduction
from quantum_k_classifier import create_qnn_multiclass_model, convert_to_circuit
from classical_k_classifier import create_fair_classical_k_model
from experiment_utils import (
    set_seed, majority_baseline, random_baseline, compute_metrics,
    paired_significance_test, holm_bonferroni, log_experiment_metadata,
    save_confusion_matrix, bootstrap_ci,
)


# ===================================================================
# Configuration
# ===================================================================

K_VALUES = [2, 3, 4, 5, 8]
N_OUTER_FOLDS = int(os.environ.get('N_FOLDS', 5))
N_INNER_FOLDS = 3
EPOCHS = 20
INNER_EPOCHS = 10
BATCH_SIZE = 32
Q_SAMPLES = int(os.environ.get('Q_SAMPLES', 400))
IS_SMOKE = os.environ.get('SMOKE_TEST', 'false').lower() == 'true'
RESULTS_DIR = os.path.join('phase5', os.environ.get('RESULTS_DIR', 'results_rigorous'))

Q_SWEEP = {
    'n_layers': [1, 2, 3],
    'learning_rate': [0.01, 0.05, 0.1],
}
C_SWEEP = {
    'hidden_units': [1, 2, 4],
    'learning_rate': [0.01, 0.05, 0.1],
}

if IS_SMOKE:
    print("!!! SMOKE TEST MODE !!!")
    K_VALUES = [2]
    N_OUTER_FOLDS = 2
    N_INNER_FOLDS = 2
    Q_SAMPLES = 20
    EPOCHS = 2
    INNER_EPOCHS = 1
    Q_SWEEP = {'n_layers': [1], 'learning_rate': [0.01]}
    C_SWEEP = {'hidden_units': [1], 'learning_rate': [0.01]}


# ===================================================================
# Inner Loop: Hyperparameter Tuning
# ===================================================================

def tune_hyperparameters(k, model_type, X_train_pca, y_train, fold_id):
    """Perform Inner CV to find best hyperparameters."""
    print(f"    [Fold {fold_id}] Tuning {model_type} hyperparameters (Inner CV)...")
    
    sweep_space = Q_SWEEP if model_type == 'quantum' else C_SWEEP
    keys, values = zip(*sweep_space.items())
    combinations = list(itertools.product(*values))
    
    inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=42)
    
    best_avg_acc = -1
    best_params = None

    for comb in combinations:
        params = dict(zip(keys, comb))
        fold_accs = []
        
        for i_train_idx, i_val_idx in inner_cv.split(X_train_pca, y_train):
            # Inner split
            Xi_tr, Xi_val = X_train_pca[i_train_idx], X_train_pca[i_val_idx]
            yi_tr, yi_val = y_train[i_train_idx], y_train[i_val_idx]
            
            set_seed(42)
            if model_type == 'quantum':
                # Convert to circuits
                xi_tr_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in Xi_tr])
                xi_val_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in Xi_val])
                model = create_qnn_multiclass_model(k, **params)
                model.fit(xi_tr_circ, yi_tr, epochs=INNER_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                acc = model.evaluate(xi_val_circ, yi_val, verbose=0)[1]
            else:
                model = create_fair_classical_k_model(k, input_shape=(16,), **params)
                model.fit(Xi_tr, yi_tr, epochs=INNER_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                acc = model.evaluate(Xi_val, yi_val, verbose=0)[1]
            
            fold_accs.append(acc)
            
        avg_acc = np.mean(fold_accs)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params = params
            
    print(f"    [Fold {fold_id}] Best {model_type} params: {best_params} (Inner Val Acc: {best_avg_acc:.4f})")
    return best_params


# ===================================================================
# Outer Loop: Performance Estimation
# ===================================================================

def run_nested_cv(k):
    """Run full nested CV for K categories."""
    print(f"\n{'='*60}")
    print(f"K={k} NESTED CV EXPERIMENT ({N_OUTER_FOLDS}x{N_INNER_FOLDS}, Q_SAMPLES={Q_SAMPLES})")
    print(f"{'='*60}")

    categories = get_top_k_categories(k)
    X_28, y = load_plankton_k_all(categories, img_size=(28, 28))
    
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)
    outer_results = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_28, y)):
        print(f"\n--- Outer Fold {fold_id} ---")
        
        # 1. Subsample training data to fixed budget (Q_SAMPLES)
        train_limit = min(len(train_idx), Q_SAMPLES)
        if train_limit < len(train_idx):
            ss = StratifiedShuffleSplit(n_splits=1, train_size=train_limit, random_state=42 + fold_id)
            sub_idx, _ = next(ss.split(train_idx, y[train_idx]))
            train_idx_limited = train_idx[sub_idx]
        else:
            train_idx_limited = train_idx

        X_tr_28, y_tr = X_28[train_idx_limited], y[train_idx_limited]
        X_te_28, y_te = X_28[test_idx], y[test_idx]

        # 2. PCA: Fit on Outer Train, Transform both
        X_tr_pca, X_te_pca, _pca = apply_pca_reduction(X_tr_28, X_te_28, n_components=16)

        # 3. Inner Loop: Tune Hyperparameters
        best_q_params = tune_hyperparameters(k, 'quantum', X_tr_pca, y_tr, fold_id)
        best_c_params = tune_hyperparameters(k, 'classical', X_tr_pca, y_tr, fold_id)

        # 4. Train with best params on full outer train set
        set_seed(42 + fold_id)
        fold_res = {'k': k, 'fold': fold_id, 'q_params': str(best_q_params), 'c_params': str(best_c_params)}

        # --- Quantum ---
        x_tr_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_tr_pca])
        x_te_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_te_pca])
        
        q_model = create_qnn_multiclass_model(k, **best_q_params)
        start = time.time()
        q_model.fit(x_tr_circ, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, 
                    validation_split=0.1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        fold_res['q_time'] = time.time() - start
        q_pred = np.argmax(q_model.predict(x_te_circ, verbose=0), axis=1)
        q_metrics = compute_metrics(y_te, q_pred, k=k)
        fold_res['q_acc'] = q_metrics['accuracy']
        fold_res['q_f1'] = q_metrics['macro_f1']

        # --- Classical ---
        c_model = create_fair_classical_k_model(k, input_shape=(16,), **best_c_params)
        start = time.time()
        c_model.fit(X_tr_pca, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                    validation_split=0.1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        fold_res['c_time'] = time.time() - start
        c_pred = np.argmax(c_model.predict(X_te_pca, verbose=0), axis=1)
        c_metrics = compute_metrics(y_te, c_pred, k=k)
        fold_res['c_acc'] = c_metrics['accuracy']
        fold_res['c_f1'] = c_metrics['macro_f1']
        
        # Baselines
        fold_res['majority_baseline'] = majority_baseline(y_te)
        fold_res['random_baseline'] = random_baseline(y_te, k=k)['analytical']

        print(f"  Fold {fold_id} Final: Q_acc={fold_res['q_acc']:.3f}, C_acc={fold_res['c_acc']:.3f}")
        
        # Save confusion matrices for each fold
        cm_dir = os.path.join(RESULTS_DIR, 'confusion_matrices', f'k{k}')
        save_confusion_matrix(q_metrics['confusion_matrix'], os.path.join(cm_dir, f'q_fold{fold_id}.csv'))
        save_confusion_matrix(c_metrics['confusion_matrix'], os.path.join(cm_dir, f'c_fold{fold_id}.csv'))

        outer_results.append(fold_res)

    return outer_results


# ===================================================================
# Main execution
# ===================================================================

def perform_rigorous_comparison():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_fold_results = []
    all_summaries = []

    for k in K_VALUES:
        try:
            fold_results = run_nested_cv(k)
            all_fold_results.extend(fold_results)

            df_k = pd.DataFrame(fold_results)
            summary = {'k': k}
            for m in ['q_acc', 'q_f1', 'c_acc', 'c_f1', 'q_time', 'c_time']:
                summary[f'{m}_mean'] = df_k[m].mean()
                summary[f'{m}_std'] = df_k[m].std()
                ci = bootstrap_ci(df_k[m].values)
                summary[f'{m}_ci_lower'] = ci['ci_lower']
                summary[f'{m}_ci_upper'] = ci['ci_upper']
            
            summary['majority_baseline'] = df_k['majority_baseline'].mean()
            summary['random_baseline'] = df_k['random_baseline'].mean()

            # Paired significance test across folds
            sig = paired_significance_test(df_k['q_acc'].values, df_k['c_acc'].values)
            summary['p_value'] = sig['p_value']
            summary['test_used'] = sig['test_used']
            
            all_summaries.append(summary)

        except Exception as e:
            print(f"ERROR on K={k}: {e}")
            import traceback; traceback.print_exc()

    # Multiple comparison correction
    if len(all_summaries) > 0:
        raw_pvals = {str(s['k']): s['p_value'] for s in all_summaries}
        corrected = holm_bonferroni(raw_pvals)
        for s in all_summaries:
            k_key = str(s['k'])
            s['corrected_p'] = corrected[k_key]['corrected_p']
            s['significant_05'] = corrected[k_key]['significant_05']

    # Final Save
    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(os.path.join(RESULTS_DIR, 'nested_cv_fold_results.csv'), index=False)
    
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'nested_cv_summary.csv'), index=False)
    
    _plot_results(df_summary)
    print(f"\nRigorous Experiment Complete. Results in {RESULTS_DIR}")

def _plot_results(df_summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    k_vals = df_summary['k'].values

    for ax, q_m, c_m, q_s, c_s, ylabel, title in [
        (ax1, 'q_acc_mean', 'c_acc_mean', 'q_acc_std', 'c_acc_std', 'Test Accuracy', 'Accuracy (Nested CV)'),
        (ax2, 'q_f1_mean', 'c_f1_mean', 'q_f1_std', 'c_f1_std', 'Macro F1-Score', 'F1-Score (Nested CV)'),
    ]:
        ax.errorbar(k_vals, df_summary[q_m], yerr=df_summary[q_s], label='Quantum (Nested CV)', marker='o', capsize=5)
        ax.errorbar(k_vals, df_summary[c_m], yerr=df_summary[c_s], label='Classical Fair (Nested CV)', marker='s', capsize=5)
        
        ax.plot(k_vals, df_summary['majority_baseline'], '--', color='gray', label='Majority', alpha=0.6)
        ax.plot(k_vals, df_summary['random_baseline'], ':', color='gray', label='Random', alpha=0.6)
        
        # Sig markers
        if 'significant_05' in df_summary.columns:
            for i, row in df_summary.iterrows():
                if row.get('significant_05', False):
                    y_max = max(row[q_m], row[c_m]) + max(row[q_s], row[c_s]) + 0.02
                    ax.annotate('*', (row['k'], y_max), ha='center', fontsize=16, fontweight='bold', color='red')

        ax.set_xlabel('K Categories')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rigorous_scientific_comparison.png'))

if __name__ == "__main__":
    perform_rigorous_comparison()
