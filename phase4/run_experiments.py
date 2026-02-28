import os

# Set threading limits via environment variables BEFORE any imports
os.environ['TF_NUM_INTRA_OP_THREADS'] = os.environ.get('TF_THREADS', '1')
os.environ['TF_NUM_INTER_OP_THREADS'] = os.environ.get('TF_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = os.environ.get('TF_THREADS', '1')

print("--- System: Warming up Quantum Pipeline (Thread Limit: {}) ---".format(os.environ['TF_NUM_INTRA_OP_THREADS']))

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

def create_fair_classical_model(input_shape=(16,)):
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
    data_qubits = cirq.GridQubit.rect(4, 4)
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
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    return tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))

def create_qnn_model(learning_rate=0.01):
    model_circuit, model_readout = create_quantum_model()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])
    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[hinge_accuracy]
    )
    return model

class CoolingCallback(tf.keras.callbacks.Callback):
    """Aggressive cooling: sleeps after batches and epochs."""
    def __init__(self, epoch_sec=1.0, batch_sec=0.0, pbar=None):
        super().__init__()
        self.epoch_sec = epoch_sec
        self.batch_sec = batch_sec
        self.pbar = pbar

    def on_train_batch_end(self, batch, logs=None):
        if self.pbar:
            self.pbar.update(1)
        if self.batch_sec > 0:
            time.sleep(self.batch_sec)

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_sec > 0:
            time.sleep(self.epoch_sec)

# --- Experiment Execution ---

def run_single_trial(class_a, class_b, trial_id, q_samples=200):
    trial_seed = 42 + trial_id
    np.random.seed(trial_seed)
    tf.random.set_seed(trial_seed)
    
    breathe_sleep = float(os.environ.get('BREATHE_SLEEP', 0.05))
    epoch_cool = float(os.environ.get('EPOCH_COOL', 1.0))
    batch_cool = float(os.environ.get('BATCH_COOL', 0.1)) # NEW: Cool after every batch
    
    trial_results = {}

    X_train_raw, X_test_raw, y_train, y_test = load_plankton_binary(class_a, class_b, img_size=(28, 28), random_state=trial_seed)
    
    X_train_cnn = X_train_raw[..., np.newaxis]
    X_test_cnn = X_test_raw[..., np.newaxis]
    
    # 1. CNN Training
    epochs_cnn = 5; batch_size = 32
    steps_per_epoch = int(np.ceil(len(X_train_cnn) / batch_size))
    with tqdm(total=epochs_cnn * steps_per_epoch, desc="    CNN Training", leave=False) as tpbar:
        cnn = create_cnn_model()
        cnn.fit(X_train_cnn, y_train, epochs=epochs_cnn, batch_size=batch_size, verbose=0, 
                callbacks=[CoolingCallback(epoch_cool, batch_cool, tpbar)])
        trial_results['cnn_acc'] = cnn.evaluate(X_test_cnn, y_test, verbose=0)[1]
    
    # 2. PCA
    X_train_pca, X_test_pca, _ = apply_pca_reduction(X_train_raw, X_test_raw, n_components=16)
    
    # 3. Fair Classical Training
    epochs_fair = 10
    steps_per_epoch = int(np.ceil(len(X_train_pca) / batch_size))
    with tqdm(total=epochs_fair * steps_per_epoch, desc="    Fair Training", leave=False) as tpbar:
        fair_nn = create_fair_classical_model()
        fair_nn.fit(X_train_pca, y_train, epochs=epochs_fair, batch_size=batch_size, verbose=0,
                    callbacks=[CoolingCallback(epoch_cool, batch_cool, tpbar)])
        trial_results['fair_acc'] = fair_nn.evaluate(X_test_pca, y_test, verbose=0)[1]

    # 4. Quantum Circuit Conv
    x_train_circ_list = []
    for x in tqdm(X_train_pca, desc="    Circuit Conv (Train)", leave=False):
        x_train_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 10.0)
        
    x_test_circ_list = []
    for x in tqdm(X_test_pca, desc="    Circuit Conv (Test)", leave=False):
        x_test_circ_list.append(convert_to_circuit(x))
        if breathe_sleep > 0: time.sleep(breathe_sleep / 10.0)

    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ_list)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ_list)
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    # 5. QNN Training
    epochs_qnn = 10; q_limit = min(len(x_train_tfcirc), q_samples)
    q_batch_size = 16 # Optimized in Phase 3
    steps_per_epoch = int(np.ceil(q_limit / q_batch_size))
    with tqdm(total=epochs_qnn * steps_per_epoch, desc="    QNN Training", leave=False) as tpbar:
        qnn = create_qnn_model()
        qnn.fit(x_train_tfcirc[:q_limit], y_train_hinge[:q_limit], epochs=epochs_qnn, batch_size=q_batch_size, verbose=0,
                callbacks=[CoolingCallback(epoch_cool, batch_cool, tpbar)])
        trial_results['qnn_acc'] = qnn.evaluate(x_test_tfcirc, y_test_hinge, verbose=0)[1]

    return trial_results

def run_experiment(class_a, class_b, num_trials=3, q_samples=200):
    print(f"\n--- Experiment: {class_a} vs {class_b} ---")
    all_trials = []
    thermal_sleep = float(os.environ.get('THERMAL_SLEEP', 0))
    
    for i in range(num_trials):
        print(f"  Trial {i+1}/{num_trials}")
        res = run_single_trial(class_a, class_b, i, q_samples=q_samples)
        all_trials.append(res)
        if thermal_sleep > 0 and i < num_trials - 1:
            print(f"  Cooling Down {thermal_sleep}s...")
            time.sleep(thermal_sleep)
    
    df_trials = pd.DataFrame(all_trials)
    summary = {'pair': f"{class_a}_vs_{class_b}"}
    for col in df_trials.columns:
        summary[f"{col}_mean"] = df_trials[col].mean()
        summary[f"{col}_std"] = df_trials[col].std()
    
    # Statistical Rigor: Welch's t-test (Equal variance not assumed)
    # Testing if QNN accuracy is significantly different from Fair Classical
    t_stat, p_val = stats.ttest_ind(df_trials['qnn_acc'], df_trials['fair_acc'], equal_var=False)
    summary['t_stat'] = t_stat
    summary['p_value'] = p_val
    summary['significant'] = p_val < 0.05
    
    print(f"  Results: QNN={summary['qnn_acc_mean']:.4f}, Fair={summary['fair_acc_mean']:.4f}")
    print(f"  P-Value: {p_val:.4f} ({'SIGNIFICANT' if summary['significant'] else 'NOT SIGNIFICANT'})")
    return summary

if __name__ == "__main__":
    pairs = [('dinobryon', 'nauplius'), ('maybe_cyano', 'diaphanosoma'), ('asterionella', 'uroglena'), ('cyclops', 'ceratium')]
    NUM_TRIALS = int(os.environ.get('NUM_TRIALS', 5))
    Q_SAMPLES = int(os.environ.get('Q_SAMPLES', 200))
    IS_SMOKE = os.environ.get('SMOKE_TEST', 'false').lower() == 'true'
    
    if IS_SMOKE:
        pairs = pairs[:1]; NUM_TRIALS = 2; Q_SAMPLES = 10

    print(f"Total Experiments: {len(pairs)} | Trials per Exp: {NUM_TRIALS} | Q_SAMPLES: {Q_SAMPLES}")
    
    all_results = []
    for a, b in pairs:
        try:
            res = run_experiment(a, b, num_trials=NUM_TRIALS, q_samples=Q_SAMPLES)
            all_results.append(res)
        except Exception as e:
            print(f"Failed experiment {a} vs {b}: {e}")

    output_dir = os.environ.get('RESULTS_DIR', 'phase4/results')
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(all_results).to_csv(os.path.join(output_dir, 'experiment_results.csv'), index=False)
