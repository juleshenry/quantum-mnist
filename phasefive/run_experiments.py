import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_plankton_k_categories, get_top_k_categories
from quantum_k_classifier import create_qnn_multiclass_model, convert_to_circuit
from classical_k_classifier import create_fair_classical_k_model, create_cnn_k_model

# Configuration
K_VALUES = [2, 3, 4, 5, 8, 12, 16]
NUM_TRIALS = 3
EPOCHS = 10
BATCH_SIZE = 32
Q_SAMPLES = 400 # Balanced limit to ensure scientifically meaningful results within time constraints

def run_k_experiment(k):
    print(f"\n### Evaluating K={k} Categories ###")
    categories = get_top_k_categories(k)
    print(f"Categories: {categories}")
    
    # Load data
    X_train_4, X_test_4, y_train, y_test = load_plankton_k_categories(categories, img_size=(4, 4))
    X_train_28, X_test_28, _, _ = load_plankton_k_categories(categories, img_size=(28, 28))
    
    # Pre-convert circuits to save time in trials
    x_train_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_train_4])
    x_test_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_test_4])
    
    # Sub-sample if necessary for QNN
    q_limit = min(len(x_train_circ), Q_SAMPLES)
    
    k_results = []
    
    for trial in range(NUM_TRIALS):
        print(f"  Trial {trial+1}/{NUM_TRIALS}...")
        trial_data = {'k': k, 'trial': trial + 1}
        
        # 1. Quantum Model
        qnn = create_qnn_multiclass_model(k)
        start = time.time()
        qnn.fit(x_train_circ[:q_limit], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        trial_data['qnn_time'] = time.time() - start
        q_pred = np.argmax(qnn.predict(x_test_circ), axis=1)
        trial_data['qnn_acc'] = np.mean(q_pred == y_test)
        trial_data['qnn_f1'] = f1_score(y_test, q_pred, average='macro')
        
        # 2. Fair Classical
        fair_nn = create_fair_classical_k_model(k)
        start = time.time()
        # Use same number of samples as QNN for fairness in training exposure
        fair_nn.fit(X_train_4[:q_limit], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        trial_data['fair_time'] = time.time() - start
        f_pred = np.argmax(fair_nn.predict(X_test_4), axis=1)
        trial_data['fair_acc'] = np.mean(f_pred == y_test)
        trial_data['fair_f1'] = f1_score(y_test, f_pred, average='macro')
        
        # 3. CNN (Baseline with full info)
        cnn = create_cnn_k_model(k)
        start = time.time()
        cnn.fit(X_train_28[:q_limit, ..., np.newaxis], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        trial_data['cnn_time'] = time.time() - start
        c_pred = np.argmax(cnn.predict(X_test_28[..., np.newaxis]), axis=1)
        trial_data['cnn_acc'] = np.mean(c_pred == y_test)
        trial_data['cnn_f1'] = f1_score(y_test, c_pred, average='macro')
        
        k_results.append(trial_data)
        
    return k_results

def plot_results(df_summary):
    plt.figure(figsize=(12, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    for model in ['qnn', 'fair', 'cnn']:
        plt.errorbar(df_summary['k'], df_summary[f'{model}_acc_mean'], 
                     yerr=df_summary[f'{model}_acc_std'], label=model.upper(), marker='o', capsize=5)
    plt.xlabel('Number of Categories (K)')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Number of Categories')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # F1 Plot
    plt.subplot(1, 2, 2)
    for model in ['qnn', 'fair', 'cnn']:
        plt.errorbar(df_summary['k'], df_summary[f'{model}_f1_mean'], 
                     yerr=df_summary[f'{model}_f1_std'], label=model.upper(), marker='s', capsize=5)
    plt.xlabel('Number of Categories (K)')
    plt.ylabel('Macro F1-Score')
    plt.title('F1-Score vs Number of Categories')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('phasefive/results/k_scaling_comparison.png')
    print("Scaling plot saved to phasefive/results/k_scaling_comparison.png")

if __name__ == "__main__":
    os.makedirs('phasefive/results', exist_ok=True)
    all_results = []
    
    for k in K_VALUES:
        res = run_k_experiment(k)
        all_results.extend(res)
    
    df = pd.DataFrame(all_results)
    df.to_csv('phasefive/results/comprehensive_k_results.csv', index=False)
    
    # Aggregate summary
    summary = df.groupby('k').agg(['mean', 'std']).reset_index()
    # Flatten multi-index columns
    summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in summary.columns]
    summary.to_csv('phasefive/results/comprehensive_k_summary.csv', index=False)
    
    print("\n--- Final Summary ---")
    print(summary[['k', 'qnn_acc_mean', 'fair_acc_mean', 'cnn_acc_mean']])
    
    plot_results(summary)
