import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_loader import load_plankton_k_categories, get_top_k_categories, apply_pca_reduction
from quantum_k_classifier import create_qnn_multiclass_model, convert_to_circuit
from classical_k_classifier import create_fair_classical_k_model, create_cnn_k_model

# Configuration
K_VALUES = [2, 3, 4, 5, 8, 12, 16]
NUM_TRIALS = 3
EPOCHS = 10
BATCH_SIZE = 32
Q_SAMPLES = 400 

class CoolingCallback(tf.keras.callbacks.Callback):
    """Sleeps between epochs to allow CPU to cool."""
    def __init__(self, seconds=1.0):
        super().__init__()
        self.seconds = seconds
    def on_epoch_end(self, epoch, logs=None):
        if self.seconds > 0:
            time.sleep(self.seconds)

def run_k_experiment(k):
    categories = get_top_k_categories(k)
    
    # Configure Pacing
    epoch_cool = float(os.environ.get('EPOCH_COOL', 1.0))
    breathe_sleep = float(os.environ.get('BREATHE_SLEEP', 0.05))

    # Load high-res data (28x28) for PCA and CNN
    X_train_raw, X_test_raw, y_train, y_test = load_plankton_k_categories(categories, img_size=(28, 28))
    
    # Apply PCA to reduce to 25 optimized features (for 5x5 QNN and Fair Classical)
    X_train_pca, X_test_pca, _ = apply_pca_reduction(X_train_raw, X_test_raw, n_components=25)
    
    # Pre-convert PCA features to circuits with breathing
    x_train_circ_list = []
    for x in X_train_pca:
        x_train_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 100.0)
        
    x_test_circ_list = []
    for x in X_test_pca:
        x_test_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 100.0)

    x_train_circ = tfq.convert_to_tensor(x_train_circ_list)
    x_test_circ = tfq.convert_to_tensor(x_test_circ_list)
    
    # Sub-sample if necessary for QNN
    q_limit = min(len(x_train_circ), Q_SAMPLES)
    
    k_results = []
    thermal_sleep = float(os.environ.get('THERMAL_SLEEP', 0))
    
    trial_pbar = tqdm(range(NUM_TRIALS), desc=f"  K={k} Trials", leave=False)
    for trial in trial_pbar:
        trial_data = {'k': k, 'trial': trial + 1}
        
        trial_pbar.set_postfix(step="Quantum")
        qnn = create_qnn_multiclass_model(k)
        start = time.time()
        qnn.fit(x_train_circ[:q_limit], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
        trial_data['qnn_time'] = time.time() - start
        q_pred = np.argmax(qnn.predict(x_test_circ), axis=1)
        trial_data['qnn_acc'] = np.mean(q_pred == y_test)
        trial_data['qnn_f1'] = f1_score(y_test, q_pred, average='macro')
        
        trial_pbar.set_postfix(step="Classical Fair")
        fair_nn = create_fair_classical_k_model(k)
        start = time.time()
        fair_nn.fit(X_train_pca[:q_limit], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
        trial_data['fair_time'] = time.time() - start
        f_pred = np.argmax(fair_nn.predict(X_test_pca), axis=1)
        trial_data['fair_acc'] = np.mean(f_pred == y_test)
        trial_data['fair_f1'] = f1_score(y_test, f_pred, average='macro')
        
        trial_pbar.set_postfix(step="CNN")
        cnn = create_cnn_k_model(k)
        start = time.time()
        cnn.fit(X_train_raw[:q_limit, ..., np.newaxis], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
        trial_data['cnn_time'] = time.time() - start
        c_pred = np.argmax(cnn.predict(X_test_raw[..., np.newaxis]), axis=1)
        trial_data['cnn_acc'] = np.mean(c_pred == y_test)
        trial_data['cnn_f1'] = f1_score(y_test, c_pred, average='macro')
        
        k_results.append(trial_data)
        if thermal_sleep > 0 and trial < NUM_TRIALS - 1:
            trial_pbar.set_postfix(step=f"Pacing {thermal_sleep}s")
            time.sleep(thermal_sleep)
        
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
    plt.title('Accuracy vs Number of Categories (5x5 PCA)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # F1 Plot
    plt.subplot(1, 2, 2)
    for model in ['qnn', 'fair', 'cnn']:
        plt.errorbar(df_summary['k'], df_summary[f'{model}_f1_mean'], 
                     yerr=df_summary[f'{model}_f1_std'], label=model.upper(), marker='s', capsize=5)
    plt.xlabel('Number of Categories (K)')
    plt.ylabel('Macro F1-Score')
    plt.title('F1-Score vs Number of Categories (5x5 PCA)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('phasefive/results/k_scaling_comparison.png')
    print("Scaling plot saved to phasefive/results/k_scaling_comparison.png")

if __name__ == "__main__":
    # Limit number of threads to prevent slamming all cores (vital for emulation/Macs)
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('TF_THREADS', 1)))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get('TF_THREADS', 1)))

    os.makedirs('phasefive/results', exist_ok=True)
    all_results = []
    
    main_pbar = tqdm(K_VALUES, desc="Scaling Study Progress")
    for k in main_pbar:
        main_pbar.set_description(f"Scaling Study K={k}")
        res = run_k_experiment(k)
        all_results.extend(res)
    
    df = pd.DataFrame(all_results)
    df.to_csv('phasefive/results/comprehensive_k_results.csv', index=False)
    
    summary = df.groupby('k').agg(['mean', 'std']).reset_index()
    summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in summary.columns]
    summary.to_csv('phasefive/results/comprehensive_k_summary.csv', index=False)
    
    print("\n--- Final Summary (5x5 PCA) ---")
    print(summary[['k', 'qnn_acc_mean', 'fair_acc_mean', 'cnn_acc_mean']])
    
    plot_results(summary)
