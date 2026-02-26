import os

# Set threading limits via environment variables BEFORE any imports
os.environ['TF_NUM_INTRA_OP_THREADS'] = os.environ.get('TF_THREADS', '1')
os.environ['TF_NUM_INTER_OP_THREADS'] = os.environ.get('TF_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = os.environ.get('TF_THREADS', '1')

print("--- System: Warming up Scientific Sweeper (Thread Limit: {}) ---".format(os.environ['TF_NUM_INTRA_OP_THREADS']))

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import itertools
from tqdm import tqdm

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
Q_SWEEP = {'n_layers': [1, 2], 'learning_rate': [0.01, 0.05]}
C_SWEEP = {'hidden_units': [1, 2], 'learning_rate': [0.01, 0.05]}

class CoolingCallback(tf.keras.callbacks.Callback):
    def __init__(self, seconds=1.0):
        super().__init__()
        self.seconds = seconds
    def on_epoch_end(self, epoch, logs=None):
        if self.seconds > 0:
            time.sleep(self.seconds)

def run_sweep(k, model_type, x_train, y_train, x_val, y_val):
    epoch_cool = float(os.environ.get('EPOCH_COOL', 1.0))
    print(f"  Sweeping {model_type} for K={k}...")
    best_acc = -1; best_params = None
    
    sweep_config = Q_SWEEP if model_type == 'quantum' else C_SWEEP
    keys, values = zip(*sweep_config.items())
    combinations = list(itertools.product(*values))
    
    for i, v in enumerate(combinations):
        params = dict(zip(keys, v))
        model = create_qnn_multiclass_model(k, **params) if model_type == 'quantum' else create_fair_classical_k_model(k, **params)
        model.fit(x_train, y_train, epochs=3, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
        acc = model.evaluate(x_val, y_val, verbose=0)[1]
        if acc > best_acc:
            best_acc = acc; best_params = params
    
    print(f"    Best {model_type} Params: {best_params} (Val Acc: {best_acc:.4f})")
    return best_params

def perform_comparison():
    # Limit number of threads to prevent slamming all cores
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('TF_THREADS', 1)))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get('TF_THREADS', 1)))

    os.makedirs('phasefive/results', exist_ok=True)
    all_trial_results = []
    epoch_cool = float(os.environ.get('EPOCH_COOL', 1.0))
    breathe_sleep = float(os.environ.get('BREATHE_SLEEP', 0.05))

    main_pbar = tqdm(K_VALUES, desc="Scientific Comparison Progress")
    for k in main_pbar:
        main_pbar.set_description(f"Scientific K={k}")
        categories = get_top_k_categories(k)
        X_train_raw, X_test_raw, y_train, y_test = load_plankton_k_categories(categories, img_size=(28, 28))
        X_train_pca, X_test_pca, _ = apply_pca_reduction(X_train_raw, X_test_raw, n_components=25)
        
        split = int(len(X_train_pca) * 0.8)
        X_tr, X_val = X_train_pca[:split], X_train_pca[split:]
        y_tr, y_val = y_train[:split], y_train[split:]
        
        print(f"  Converting circuits for K={k}...")
        X_tr_q_list = []
        for x in X_tr[:Q_SAMPLES]:
            X_tr_q_list.append(convert_to_circuit(x))
            if breathe_sleep > 0: time.sleep(breathe_sleep / 100.0)
            
        X_val_q_list = []
        for x in X_val:
            X_val_q_list.append(convert_to_circuit(x))
            if breathe_sleep > 0: time.sleep(breathe_sleep / 100.0)

        X_tr_q = tfq.convert_to_tensor(X_tr_q_list)
        X_val_q = tfq.convert_to_tensor(X_val_q_list)
        X_test_q = tfq.convert_to_tensor([convert_to_circuit(x) for x in X_test_pca])
        y_tr_q = y_tr[:Q_SAMPLES]
        
        best_q_params = run_sweep(k, 'quantum', X_tr_q, y_tr_q, X_val_q, y_val)
        best_c_params = run_sweep(k, 'classical', X_tr[:Q_SAMPLES], y_tr_q, X_val, y_val)
        
        thermal_sleep = float(os.environ.get('THERMAL_SLEEP', 0))
        trial_pbar = tqdm(range(NUM_TRIALS), desc=f"  Trials K={k}", leave=False)
        for t in trial_pbar:
            trial_res = {'k': k, 'trial': t+1}
            
            trial_pbar.set_postfix(step="Quantum")
            q_model = create_qnn_multiclass_model(k, **best_q_params)
            q_model.fit(X_tr_q, y_tr_q, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
            q_pred = np.argmax(q_model.predict(X_test_q), axis=1)
            trial_res['q_acc'] = np.mean(q_pred == y_test); trial_res['q_f1'] = f1_score(y_test, q_pred, average='macro')
            
            trial_pbar.set_postfix(step="Classical")
            c_model = create_fair_classical_k_model(k, **best_c_params)
            c_model.fit(X_tr[:Q_SAMPLES], y_tr_q, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
            c_pred = np.argmax(c_model.predict(X_test_pca), axis=1)
            trial_res['c_acc'] = np.mean(c_pred == y_test); trial_res['c_f1'] = f1_score(y_test, c_pred, average='macro')
            
            all_trial_results.append(trial_res)
            if thermal_sleep > 0 and t < NUM_TRIALS - 1:
                trial_pbar.set_postfix(step=f"Pacing {thermal_sleep}s")
                time.sleep(thermal_sleep)
            
    df = pd.DataFrame(all_trial_results); df.to_csv('phasefive/results/scientific_k_comparison.csv', index=False)
    summary = df.groupby('k').agg(['mean', 'std']).reset_index()
    summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in summary.columns]
    summary.to_csv('phasefive/results/scientific_k_summary.csv', index=False)

if __name__ == "__main__":
    perform_comparison()
