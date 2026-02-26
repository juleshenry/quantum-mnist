import os
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import time
import pandas as pd
from data_loader import load_plankton_binary

# --- Models ---

def create_fair_classical_model(input_shape=(4, 4, 1)):
    # 3 hidden units gives approx 55 parameters, matching QNN's ~48 parameters more closely
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(3, activation='relu'),
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
    
    # Add entanglement between data qubits
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

def convert_to_circuit(image):
    # Flatten the image (expects H, W or H, W, 1)
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        # Angle encoding
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

# --- Experiment Execution ---

def run_single_trial(class_a, class_b, trial_id, q_samples=200):
    print(f"  Trial {trial_id+1}...")
    trial_results = {}

    # 1. Classical CNN (28x28)
    X_train, X_test, y_train, y_test = load_plankton_binary(class_a, class_b, img_size=(28, 28))
    if len(X_train.shape) == 3:
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    
    cnn = create_cnn_model()
    start = time.time()
    cnn.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    trial_results['cnn_time'] = time.time() - start
    trial_results['cnn_acc'] = cnn.evaluate(X_test, y_test, verbose=0)[1]
    
    # 2. Fair Classical (4x4)
    X_train_4, X_test_4, y_train_4, y_test_4 = load_plankton_binary(class_a, class_b, img_size=(4, 4))
    if len(X_train_4.shape) == 3:
        X_train_4 = X_train_4[..., np.newaxis]
        X_test_4 = X_test_4[..., np.newaxis]
    
    fair_nn = create_fair_classical_model()
    start = time.time()
    fair_nn.fit(X_train_4, y_train_4, epochs=10, batch_size=32, verbose=0)
    trial_results['fair_time'] = time.time() - start
    trial_results['fair_acc'] = fair_nn.evaluate(X_test_4, y_test_4, verbose=0)[1]

    # 3. Quantum Model (4x4)
    x_train_circ = [convert_to_circuit(x) for x in X_train_4]
    x_test_circ = [convert_to_circuit(x) for x in X_test_4]
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
    y_train_hinge = 2.0 * y_train_4 - 1.0
    y_test_hinge = 2.0 * y_test_4 - 1.0

    qnn = create_qnn_model()
    start = time.time()
    # Use specified number of samples for QNN training
    q_limit = min(len(x_train_tfcirc), q_samples)
    qnn.fit(x_train_tfcirc[:q_limit], y_train_hinge[:q_limit], epochs=10, batch_size=32, verbose=0)
    trial_results['qnn_time'] = time.time() - start
    trial_results['qnn_acc'] = qnn.evaluate(x_test_tfcirc, y_test_hinge, verbose=0)[1]

    return trial_results

def run_experiment(class_a, class_b, num_trials=3):
    print(f"\n--- Running Experiment: {class_a} vs {class_b} ({num_trials} trials) ---")
    
    all_trials = []
    for i in range(num_trials):
        res = run_single_trial(class_a, class_b, i)
        all_trials.append(res)
    
    # Aggregate results
    df_trials = pd.DataFrame(all_trials)
    summary = {'pair': f"{class_a}_vs_{class_b}"}
    
    for col in df_trials.columns:
        summary[f"{col}_mean"] = df_trials[col].mean()
        summary[f"{col}_std"] = df_trials[col].std()
    
    print(f"Summary Results: {summary}")
    return summary

if __name__ == "__main__":
    pairs = [
        ('dinobryon', 'nauplius'),
        ('maybe_cyano', 'diaphanosoma'),
        ('asterionella', 'uroglena'),
        ('cyclops', 'ceratium')
    ]
    
    # Allow overriding via environment variables for faster runs in CI/Emulated environments
    NUM_TRIALS = int(os.environ.get('NUM_TRIALS', 3))
    Q_SAMPLES = int(os.environ.get('Q_SAMPLES', 200))
    
    print(f"Starting experiments with NUM_TRIALS={NUM_TRIALS}, Q_SAMPLES={Q_SAMPLES}")
    
    all_results = []
    # If we are in a "smoke test" mode, just run the first pair
    run_pairs = pairs if os.environ.get('FULL_RUN', 'true') == 'true' else pairs[:1]
    
    for a, b in run_pairs:
        try:
            # Pass Q_SAMPLES to run_experiment if needed, but run_experiment calls run_single_trial
            # Let's modify run_experiment to accept q_samples
            def run_experiment_with_params(ca, cb, n_trials, q_samps):
                print(f"\n--- Running Experiment: {ca} vs {cb} ({n_trials} trials, {q_samps} q_samples) ---")
                all_trials = []
                for i in range(n_trials):
                    res = run_single_trial(ca, cb, i, q_samples=q_samps)
                    all_trials.append(res)
                
                df_trials = pd.DataFrame(all_trials)
                summary = {'pair': f"{ca}_vs_{cb}"}
                for col in df_trials.columns:
                    summary[f"{col}_mean"] = df_trials[col].mean()
                    summary[f"{col}_std"] = df_trials[col].std()
                print(f"Summary Results: {summary}")
                return summary

            res = run_experiment_with_params(a, b, NUM_TRIALS, Q_SAMPLES)
            all_results.append(res)
        except Exception as e:
            print(f"Failed experiment {a} vs {b}: {e}")

    os.makedirs('phasefour/results', exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv('phasefour/results/experiment_results.csv', index=False)
    print(f"\nAll experiments completed. Results saved to phasefour/results/experiment_results.csv")
