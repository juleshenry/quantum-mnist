import os

# Set threading limits via environment variables BEFORE any imports
os.environ['TF_NUM_INTRA_OP_THREADS'] = os.environ.get('TF_THREADS', '1')
os.environ['TF_NUM_INTER_OP_THREADS'] = os.environ.get('TF_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = os.environ.get('TF_THREADS', '1')

print("--- System: Warming up Scaling Pipeline (Thread Limit: {}) ---".format(os.environ['TF_NUM_INTRA_OP_THREADS']))

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
    def __init__(self, epoch_sec=1.0, batch_sec=0.0, pbar=None):
        super().__init__()
        self.epoch_sec = epoch_sec
        self.batch_sec = batch_sec
        self.pbar = pbar
    def on_train_batch_end(self, batch, logs=None):
        if self.pbar: self.pbar.update(1)
        if self.batch_sec > 0: time.sleep(self.batch_sec)
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_sec > 0: time.sleep(self.epoch_sec)

def run_k_experiment(k):
    categories = get_top_k_categories(k)
    epoch_cool = float(os.environ.get('EPOCH_COOL', 1.0))
    batch_cool = float(os.environ.get('BATCH_COOL', 0.1))
    breathe_sleep = float(os.environ.get('BREATHE_SLEEP', 0.05))

    X_train_raw, X_test_raw, y_train, y_test = load_plankton_k_categories(categories, img_size=(28, 28))
    X_train_pca, X_test_pca, _ = apply_pca_reduction(X_train_raw, X_test_raw, n_components=16)
    
    x_train_circ_list = []
    for x in tqdm(X_train_pca, desc="    Circuit Conv (Train)", leave=False):
        x_train_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 100.0)
        
    x_test_circ_list = []
    for x in tqdm(X_test_pca, desc="    Circuit Conv (Test)", leave=False):
        x_test_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 100.0)

    x_train_circ = tfq.convert_to_tensor(x_train_circ_list)
    x_test_circ = tfq.convert_to_tensor(x_test_circ_list)
    q_limit = min(len(x_train_circ), Q_SAMPLES)
    
    k_results = []
    thermal_sleep = float(os.environ.get('THERMAL_SLEEP', 0))
    
    for trial in range(NUM_TRIALS):
        print(f"  Trial {trial+1}/{NUM_TRIALS}")
        trial_data = {'k': k, 'trial': trial + 1}
        
        # Quantum
        steps = int(np.ceil(q_limit / BATCH_SIZE))
        with tqdm(total=EPOCHS * steps, desc="    Quantum Training", leave=False) as tpbar:
            qnn = create_qnn_multiclass_model(k)
            qnn.fit(x_train_circ[:q_limit], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool, batch_cool, tpbar)])
            q_pred = np.argmax(qnn.predict(x_test_circ), axis=1)
            trial_data['qnn_acc'] = np.mean(q_pred == y_test); trial_data['qnn_f1'] = f1_score(y_test, q_pred, average='macro')
        
        # Fair Classical
        steps = int(np.ceil(q_limit / BATCH_SIZE))
        with tqdm(total=EPOCHS * steps, desc="    Fair Training", leave=False) as tpbar:
            fair_nn = create_fair_classical_k_model(k)
            fair_nn.fit(X_train_pca[:q_limit], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool, batch_cool, tpbar)])
            f_pred = np.argmax(fair_nn.predict(X_test_pca), axis=1)
            trial_data['fair_acc'] = np.mean(f_pred == y_test); trial_data['fair_f1'] = f1_score(y_test, f_pred, average='macro')
        
        # CNN
        steps = int(np.ceil(q_limit / BATCH_SIZE))
        with tqdm(total=EPOCHS * steps, desc="    CNN Training", leave=False) as tpbar:
            cnn = create_cnn_k_model(k)
            cnn.fit(X_train_raw[:q_limit, ..., np.newaxis], y_train[:q_limit], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool, batch_cool, tpbar)])
            c_pred = np.argmax(cnn.predict(X_test_raw[..., np.newaxis]), axis=1)
            trial_data['cnn_acc'] = np.mean(c_pred == y_test); trial_data['cnn_f1'] = f1_score(y_test, c_pred, average='macro')
        
        k_results.append(trial_data)
        if thermal_sleep > 0 and trial < NUM_TRIALS - 1:
            print(f"  Pacing {thermal_sleep}s..."); time.sleep(thermal_sleep)
    return k_results

def plot_results(df_summary):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for model in ['qnn', 'fair', 'cnn']:
        plt.errorbar(df_summary['k'], df_summary[f'{model}_acc_mean'], yerr=df_summary[f'{model}_acc_std'], label=model.upper(), marker='o', capsize=5)
    plt.xlabel('Number of Categories (K)'); plt.ylabel('Test Accuracy'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(1, 2, 2)
    for model in ['qnn', 'fair', 'cnn']:
        plt.errorbar(df_summary['k'], df_summary[f'{model}_f1_mean'], yerr=df_summary[f'{model}_f1_std'], label=model.upper(), marker='s', capsize=5)
    plt.xlabel('Number of Categories (K)'); plt.ylabel('Macro F1-Score'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.savefig('phase5/results/k_scaling_comparison.png')

if __name__ == "__main__":
    os.makedirs('phase5/results', exist_ok=True)
    all_results = []
    main_pbar = tqdm(K_VALUES, desc="Scaling Study Progress")
    for k in main_pbar:
        main_pbar.set_description(f"Scaling Study K={k}")
        res = run_k_experiment(k)
        all_results.extend(res)
    df = pd.DataFrame(all_results); df.to_csv('phase5/results/comprehensive_k_results.csv', index=False)
    summary = df.groupby('k').agg(['mean', 'std']).reset_index()
    summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in summary.columns]
    summary.to_csv('phase5/results/comprehensive_k_summary.csv', index=False)
    plot_results(summary)
