import os
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import time
import pandas as pd
from scipy import stats
from tqdm import tqdm
from data_loader import load_plankton_binary, apply_pca_reduction

# --- Models ---

def create_fair_classical_model(input_shape=(25,)):
    # 25-2-1 architecture gives:
    # Weights: 25*2 = 50, Bias: 2, Layer 2 weights: 2*1 = 2, Bias: 1
    # Total: 55 parameters, matching 5x5 QNN's ~50 parameters
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

def create_cnn_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Quantum Model Setup ---

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def create_quantum_model():
    # 5x5 Grid (25 data qubits)
    data_qubits = cirq.GridQubit.rect(5, 5)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()
    
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i+1]))
        
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    builder.add_layer(circuit, cirq.YY, "yy1")
    circuit.append(cirq.H(readout))
    
    return circuit, cirq.Z(readout)

def convert_to_circuit(pca_features):
    values = np.ndarray.flatten(pca_features)
    qubits = cirq.GridQubit.rect(5, 5)
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
        metrics=[hinge_accuracy]
    )
    return model

class CoolingCallback(tf.keras.callbacks.Callback):
    """Sleeps between epochs to allow CPU to cool."""
    def __init__(self, seconds=1.0):
        super().__init__()
        self.seconds = seconds
    def on_epoch_end(self, epoch, logs=None):
        if self.seconds > 0:
            time.sleep(self.seconds)

# --- Experiment Execution ---

def run_single_trial(class_a, class_b, trial_id, q_samples=200, pbar=None):
    if pbar: pbar.set_postfix(step="Data Load")
    trial_seed = 42 + trial_id
    np.random.seed(trial_seed)
    tf.random.set_seed(trial_seed)
    
    # Configure Pacing
    thermal_sleep = float(os.environ.get('THERMAL_SLEEP', 0))
    breathe_sleep = float(os.environ.get('BREATHE_SLEEP', 0.05)) # Sleep between batches/circuit convs
    epoch_cool = float(os.environ.get('EPOCH_COOL', 1.0)) # Sleep between epochs
    
    trial_results = {}

    # 1. Classical CNN (28x28)
    X_train_raw, X_test_raw, y_train, y_test = load_plankton_binary(class_a, class_b, img_size=(28, 28), random_state=trial_seed)
    
    X_train_cnn = X_train_raw[..., np.newaxis]
    X_test_cnn = X_test_raw[..., np.newaxis]
    
    if pbar: pbar.set_postfix(step="Train CNN")
    cnn = create_cnn_model()
    start = time.time()
    cnn.fit(X_train_cnn, y_train, epochs=5, batch_size=32, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
    trial_results['cnn_time'] = time.time() - start
    trial_results['cnn_acc'] = cnn.evaluate(X_test_cnn, y_test, verbose=0)[1]
    
    # 2. PCA Reduction (5x5 / 25 components)
    if pbar: pbar.set_postfix(step="PCA")
    X_train_pca, X_test_pca, _ = apply_pca_reduction(X_train_raw, X_test_raw, n_components=25)
    
    # 3. Fair Classical (Matched to 25 inputs)
    if pbar: pbar.set_postfix(step="Train Fair")
    fair_nn = create_fair_classical_model()
    start = time.time()
    fair_nn.fit(X_train_pca, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
    trial_results['fair_time'] = time.time() - start
    trial_results['fair_acc'] = fair_nn.evaluate(X_test_pca, y_test, verbose=0)[1]

    # 4. Quantum Model (5x5 grid PCA)
    if pbar: pbar.set_postfix(step="Circuit Conv")
    x_train_circ_list = []
    for x in X_train_pca:
        x_train_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 10.0) # Tiny breather
        
    x_test_circ_list = []
    for x in X_test_pca:
        x_test_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 10.0)

    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ_list)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ_list)
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    if pbar: pbar.set_postfix(step="Train QNN")
    qnn = create_qnn_model()
    start = time.time()
    q_limit = min(len(x_train_tfcirc), q_samples)
    qnn.fit(x_train_tfcirc[:q_limit], y_train_hinge[:q_limit], epochs=10, batch_size=32, verbose=0, callbacks=[CoolingCallback(epoch_cool)])
    trial_results['qnn_time'] = time.time() - start
    trial_results['qnn_acc'] = qnn.evaluate(x_test_tfcirc, y_test_hinge, verbose=0)[1]

    return trial_results

def run_experiment(class_a, class_b, num_trials=3, q_samples=200):
    print(f"\n--- Running Experiment: {class_a} vs {class_b} ---")
    
    all_trials = []
    thermal_sleep = float(os.environ.get('THERMAL_SLEEP', 0))
    
    trial_pbar = tqdm(range(num_trials), desc=f"  Trials", leave=False)
    for i in trial_pbar:
        res = run_single_trial(class_a, class_b, i, q_samples=q_samples, pbar=trial_pbar)
        all_trials.append(res)
        if thermal_sleep > 0 and i < num_trials - 1:
            trial_pbar.set_postfix(step=f"Pacing {thermal_sleep}s")
            time.sleep(thermal_sleep)
    
    df_trials = pd.DataFrame(all_trials)
    summary = {'pair': f"{class_a}_vs_{class_b}"}
    
    for col in df_trials.columns:
        summary[f"{col}_mean"] = df_trials[col].mean()
        summary[f"{col}_std"] = df_trials[col].std()
    
    t_stat, p_val = stats.ttest_ind(df_trials['qnn_acc'], df_trials['fair_acc'], equal_var=False)
    summary['p_value_qnn_vs_fair'] = p_val
    summary['significant_05'] = p_val < 0.05

    print(f"Summary Results: QNN_Acc={summary['qnn_acc_mean']:.4f}, Fair_Acc={summary['fair_acc_mean']:.4f}")
    return summary

if __name__ == "__main__":
    # --- Thermal Protection Configuration ---
    # Limit number of threads to prevent slamming all cores (vital for emulation/Macs)
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('TF_THREADS', 1)))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get('TF_THREADS', 1)))
    pairs = [
        ('dinobryon', 'nauplius'),
        ('maybe_cyano', 'diaphanosoma'),
        ('asterionella', 'uroglena'),
        ('cyclops', 'ceratium')
    ]
    
    NUM_TRIALS = int(os.environ.get('NUM_TRIALS', 3))
    Q_SAMPLES = int(os.environ.get('Q_SAMPLES', 200))
    IS_SMOKE = os.environ.get('SMOKE_TEST', 'false').lower() == 'true'
    
    if IS_SMOKE:
        print("!!! SMOKE TEST MODE ENABLED !!!")
        pairs = pairs[:1]
        NUM_TRIALS = 2
        Q_SAMPLES = 10

    print(f"Starting experiments with NUM_TRIALS={NUM_TRIALS}, Q_SAMPLES={Q_SAMPLES}")
    
    all_results = []
    main_pbar = tqdm(pairs, desc="Overall Progress")
    for a, b in main_pbar:
        main_pbar.set_description(f"Experiment: {a} vs {b}")
        try:
            res = run_experiment(a, b, num_trials=NUM_TRIALS, q_samples=Q_SAMPLES)
            all_results.append(res)
        except Exception as e:
            print(f"Failed experiment {a} vs {b}: {e}")

    output_dir = os.environ.get('RESULTS_DIR', 'phasefour/results')
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'experiment_results.csv'), index=False)
    print(f"\nAll experiments completed. Results saved to {output_dir}/experiment_results.csv")
