import os
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Import components from phase5
import sys
sys.path.append('phase5')
from data_loader import load_plankton_k_categories, get_top_k_categories, apply_pca_reduction

def create_saliency_circuit(n_features, k, n_layers=1):
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    
    # Use names that sort predictably: f00, f01, ... and w00, w01, ...
    feat_symbols = [sympy.Symbol(f'f{i:02d}') for i in range(n_features)]
    model_symbols = [sympy.Symbol(f'w{i:02d}') for i in range(2 * n_layers * len(data_qubits))]
    
    circuit = cirq.Circuit()
    # Encoding Layer
    for i, sym in enumerate(feat_symbols):
        circuit.append(cirq.ry(np.pi * sym)(data_qubits[i]))
    
    # Entanglement Layer
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i+1]))
    circuit.append(cirq.CZ(data_qubits[-1], readout))
    
    # Variational Layers
    for l in range(n_layers):
        for i, q in enumerate(data_qubits):
            idx = 2 * (l * len(data_qubits) + i)
            circuit.append(cirq.XX(q, readout)**model_symbols[idx])
            circuit.append(cirq.ZZ(q, readout)**model_symbols[idx+1])
            
    # We need k observables for k-class classification
    all_qubits = [readout] + data_qubits
    observables = [cirq.Z(all_qubits[i]) for i in range(k)]
            
    return circuit, feat_symbols, model_symbols, observables

class QuantumSaliencyModel(tf.keras.Model):
    def __init__(self, circuit, n_feat, n_model, observables):
        super().__init__()
        self.pqc = tfq.layers.ControlledPQC(circuit, observables)
        self.n_feat = n_feat
        self.n_model = n_model
        # Initialize model weights
        self.q_weights = tf.Variable(
            tf.random.normal((1, n_model), stddev=0.1),
            name='quantum_weights',
            trainable=True
        )
        self.softmax = tf.keras.layers.Softmax()
        self.base_circuit = tfq.convert_to_tensor([circuit])
        
    def call(self, feat_inputs):
        batch_size = tf.shape(feat_inputs)[0]
        tiled_circuits = tf.tile(self.base_circuit, [batch_size])
        tiled_weights = tf.tile(self.q_weights, [batch_size, 1])
        # Join features and weights: [f00, f01, ..., w00, w01, ...]
        # Note: ControlledPQC sorts symbols alphabetically. 
        # Our naming (fXX, wXX) ensures features come before weights.
        combined_params = tf.concat([feat_inputs, tiled_weights], axis=1)
        expectations = self.pqc([tiled_circuits, combined_params])
        return self.softmax(expectations)

def run_saliency_demo():
    print("--- Phase 6: Quantum Interpretability ---")
    os.makedirs('phase6/results', exist_ok=True)
    
    # 1. Load Data
    k = 2
    categories = get_top_k_categories(k)
    print(f"Categories: {categories}")
    X_train_raw, X_test_raw, y_train, y_test = load_plankton_k_categories(categories, img_size=(28, 28))
    X_train_pca, X_test_pca, pca = apply_pca_reduction(X_train_raw, X_test_raw, n_components=16)
    
    # 2. Build Model
    n_features = 16
    n_layers = 1
    circuit, f_syms, w_syms, observables = create_saliency_circuit(n_features, k, n_layers)
    model = QuantumSaliencyModel(circuit, len(f_syms), len(w_syms), observables)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # 3. Train briefly
    print("Training model on subset...")
    subset_size = 200
    model.fit(X_train_pca[:subset_size], y_train[:subset_size], epochs=5, batch_size=32, verbose=1)
    
    # 4. Compute Saliency for a few examples
    n_examples = 5
    print(f"Generating {n_examples} saliency maps...")
    for i in range(n_examples):
        img_idx = i
        x_pca = X_test_pca[img_idx:img_idx+1]
        x_raw = X_test_raw[img_idx]
        true_label = y_test[img_idx]
        
        # Predicted label
        preds = model.predict(x_pca, verbose=0)
        pred_label = np.argmax(preds[0])
        
        # Gradient
        feat_tensor = tf.convert_to_tensor(x_pca, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(feat_tensor)
            prediction = model(feat_tensor)
            # Focus on the predicted class score
            score = prediction[0, pred_label]
            
        grads = tape.gradient(score, feat_tensor).numpy()[0]
        
        # 5. Map back to image space
        # PCA Inverse: Saliency_Image = Grads * PCA_Components
        saliency_raw = np.dot(grads, pca.components_)
        saliency_img = saliency_raw.reshape(28, 28)
        
        # Normalize for visualization
        saliency_img = np.abs(saliency_img)
        if np.max(saliency_img) > 0:
            saliency_img /= np.max(saliency_img)
            
        # 6. Plotting
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(x_raw, cmap='gray')
        plt.title(f"Original ({categories[true_label]})")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        sns.heatmap(saliency_img, cmap='viridis', cbar=False)
        plt.title("Quantum Saliency Map")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(x_raw, cmap='gray')
        plt.imshow(saliency_img, cmap='hot', alpha=0.5)
        plt.title(f"Overlay (Pred: {categories[pred_label]})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'phase6/results/saliency_example_{i}.png')
        plt.close()
        
    print(f"Saved saliency maps to phase6/results/")

if __name__ == "__main__":
    run_saliency_demo()
