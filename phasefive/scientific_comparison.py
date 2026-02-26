import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import itertools

from data_loader import load_plankton_k_categories, get_top_k_categories, apply_pca_reduction
from quantum_k_classifier import create_qnn_multiclass_model, convert_to_circuit
from classical_k_classifier import create_fair_classical_k_model

# Constants
K_VALUES = [2, 3, 4, 5]
NUM_TRIALS = 3
EPOCHS = 10
BATCH_SIZE = 32
Q_SAMPLES = 400

# Sweep Spaces
Q_SWEEP = {
    'n_layers': [1, 2],
    'learning_rate': [0.01, 0.05]
}
C_SWEEP = {
    'hidden_units': [1, 2],
    'learning_rate': [0.01, 0.05]
}

def run_sweep(k, model_type, x_train, y_train, x_val, y_val):
    print(f"  Sweeping {model_type} for K={k}...")
    best_acc = -1
    best_params = None
    
    if model_type == 'quantum':
        keys, values = zip(*Q_SWEEP.items())
        combinations = list(itertools.product(*values))
        for i, v in enumerate(combinations):
            params = dict(zip(keys, v))
            print(f"    - [{i+1}/{len(combinations)}] Testing {params}...")
            model = create_qnn_multiclass_model(k, **params)
            model.fit(x_train, y_train, epochs=3, batch_size=BATCH_SIZE, verbose=0)
            acc = model.evaluate(x_val, y_val, verbose=0)[1]
            if acc > best_acc:
                best_acc = acc
                best_params = params
    else:
        keys, values = zip(*C_SWEEP.items())
        combinations = list(itertools.product(*values))
        for i, v in enumerate(combinations):
            params = dict(zip(keys, v))
            print(f"    - [{i+1}/{len(combinations)}] Testing {params}...")
            model = create_fair_classical_k_model(k, **params)
            model.fit(x_train, y_train, epochs=3, batch_size=BATCH_SIZE, verbose=0)
            acc = model.evaluate(x_val, y_val, verbose=0)[1]
            if acc > best_acc:
                best_acc = acc
                best_params = params
    
    print(f"    Best {model_type} Params: {best_params} (Val Acc: {best_acc:.4f})")
    return best_params

def perform_comparison():
    os.makedirs('phasefive/results', exist_ok=True)
    all_trial_results = []
    
    for k in K_VALUES:
        print(f"\n--- Scientific Comparison for K={k} (5x5 PCA) ---")
        categories = get_top_k_categories(k)
        # Load high-res (28x28) for PCA
        X_train_raw, X_test_raw, y_train, y_test = load_plankton_k_categories(categories, img_size=(28, 28))
        
        # PCA to 25 components
        X_train_pca, X_test_pca, _ = apply_pca_reduction(X_train_raw, X_test_raw, n_components=25)
        
        # Further split train for validation in sweep
        split = int(len(X_train_pca) * 0.8)
        X_tr, X_val = X_train_pca[:split], X_train_pca[split:]
        y_tr, y_val = y_train[:split], y_train[split:]
        
        # Prepare Quantum Circuits
        print("  Converting circuits...")
        X_tr_q = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_tr[:Q_SAMPLES]])
        X_val_q = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_val])
        X_test_q = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_test_pca])
        y_tr_q = y_tr[:Q_SAMPLES]
        
        # 1. SWEEP
        best_q_params = run_sweep(k, 'quantum', X_tr_q, y_tr_q, X_val_q, y_val)
        best_c_params = run_sweep(k, 'classical', X_tr[:Q_SAMPLES], y_tr_q, X_val, y_val)
        
        # 2. TRIALS
        thermal_sleep = float(os.environ.get('THERMAL_SLEEP', 0))
        for t in range(NUM_TRIALS):
            print(f"  Trial {t+1}/{NUM_TRIALS}...")
            trial_res = {'k': k, 'trial': t+1}
            
            # ... training logic ...
            # Quantum Trial
            print("    Training Quantum...")
            q_model = create_qnn_multiclass_model(k, **best_q_params)
            q_model.fit(X_tr_q, y_tr_q, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            q_pred = np.argmax(q_model.predict(X_test_q), axis=1)
            trial_res['q_acc'] = np.mean(q_pred == y_test)
            trial_res['q_f1'] = f1_score(y_test, q_pred, average='macro')
            
            # Classical Trial
            print("    Training Classical...")
            c_model = create_fair_classical_k_model(k, **best_c_params)
            c_model.fit(X_tr[:Q_SAMPLES], y_tr_q, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            c_pred = np.argmax(c_model.predict(X_test_pca), axis=1)
            trial_res['c_acc'] = np.mean(c_pred == y_test)
            trial_res['c_f1'] = f1_score(y_test, c_pred, average='macro')
            
            all_trial_results.append(trial_res)
            print(f"    Trial {t+1} Results: Q_Acc={trial_res['q_acc']:.4f}, C_Acc={trial_res['c_acc']:.4f}")

            if thermal_sleep > 0 and t < NUM_TRIALS - 1:
                print(f"    Pacing... sleeping for {thermal_sleep}s.")
                time.sleep(thermal_sleep)
            
    # Aggregate and Save
    df = pd.DataFrame(all_trial_results)
    df.to_csv('phasefive/results/scientific_k_comparison.csv', index=False)
    
    summary = df.groupby('k').agg(['mean', 'std']).reset_index()
    summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in summary.columns]
    summary.to_csv('phasefive/results/scientific_k_summary.csv', index=False)
    
    # Graphing
    plt.figure(figsize=(10, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.errorbar(summary['k'], summary['q_acc_mean'], yerr=summary['q_acc_std'], label='Quantum', marker='o', capsize=5)
    plt.errorbar(summary['k'], summary['c_acc_mean'], yerr=summary['c_acc_std'], label='Classical Fair', marker='s', capsize=5)
    plt.xlabel('Number of Categories (K)')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy Comparison (Swept 5x5 PCA)')
    plt.xticks(K_VALUES)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Plot
    plt.subplot(1, 2, 2)
    plt.errorbar(summary['k'], summary['q_f1_mean'], yerr=summary['q_f1_std'], label='Quantum', marker='o', capsize=5)
    plt.errorbar(summary['k'], summary['c_f1_mean'], yerr=summary['c_f1_std'], label='Classical Fair', marker='s', capsize=5)
    plt.xlabel('Number of Categories (K)')
    plt.ylabel('Macro F1-Score')
    plt.title('F1-Score Comparison (Swept 5x5 PCA)')
    plt.xticks(K_VALUES)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phasefive/results/scientific_scaling_plot.png')
    print("\nResults and Graph saved to phasefive/results/")

if __name__ == "__main__":
    perform_comparison()
