import os
import sys
import sympy
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class UnifiedQuantumClassifier:
    def __init__(self, image_size=(4, 4), encoding='angle', threshold=0.5):
        self.image_size = image_size
        self.n_qubits = image_size[0] * image_size[1]
        self.encoding = encoding
        self.threshold = threshold
        self.qubits = cirq.GridQubit.rect(*self.image_size)
        self.readout_qubit = cirq.GridQubit(-1, -1)
        print("Initialized Unified Classifier: {}x{} images, {} qubits".format(image_size[0], image_size[1], self.n_qubits))

    def preprocess_image(self, path):
        """Load and normalize image at higher resolution for PCA/Pooling."""
        try:
            # Load at 16x16 to keep some detail before compression
            img = Image.open(path).convert("L").resize((16, 16), Image.BILINEAR)
            img_array = np.array(img) / 255.0
            return img_array.flatten()
        except Exception as e:
            return None

    def to_circuit(self, features):
        circuit = cirq.Circuit()
        # Ensure we only use n_qubits
        vals = features[:self.n_qubits]
        if self.encoding == 'angle':
            for i, q in enumerate(self.qubits):
                angle = vals[i] * np.pi
                circuit.append(cirq.ry(angle)(q))
        return circuit

    def build_model(self):
        circuit = cirq.Circuit()
        circuit.append(cirq.H(self.readout_qubit))
        # Variational layers with more entanglement
        for i, q in enumerate(self.qubits):
            circuit.append(cirq.rx(sympy.Symbol("x_" + str(i)))(q))
            circuit.append(cirq.ry(sympy.Symbol("y_" + str(i)))(q))
        
        # Ring entanglement
        for i in range(self.n_qubits):
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[(i+1)%self.n_qubits]))
        
        # Connect to readout
        for i in range(0, self.n_qubits, 4): # Sample some qubits to readout
            circuit.append(cirq.CNOT(self.qubits[i], self.readout_qubit))

        readout_op = cirq.Z(self.readout_qubit)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(circuit, readout_op, differentiator=tfq.differentiators.ParameterShift()),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, cat_a, cat_b, data_dir, epochs=20):
        def load_data(cat):
            p = os.path.join(data_dir, cat, "training_data")
            if not os.path.exists(p): p = os.path.join(data_dir, cat)
            files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith('.jpeg')][:60]
            return [self.preprocess_image(f) for f in files if self.preprocess_image(f) is not None]

        print("Loading {} and {}...".format(cat_a, cat_b))
        raw_a = load_data(cat_a)
        raw_b = load_data(cat_b)
        X_raw = np.array(raw_a + raw_b)
        y = np.array([0]*len(raw_a) + [1]*len(raw_b))

        # PCA to compress 256 (16x16) features to n_qubits
        print("Compressing features via PCA (256 -> {})...".format(self.n_qubits))
        pca = PCA(n_components=self.n_qubits)
        X_compressed = pca.fit_transform(X_raw)
        # Normalize to [0, 1] for angle encoding
        X_compressed = (X_compressed - X_compressed.min()) / (X_compressed.max() - X_compressed.min())

        X_train, X_test, y_train, y_test = train_test_split(X_compressed, y, test_size=0.2, stratify=y)
        
        x_train_tf = tfq.convert_to_tensor([self.to_circuit(x) for x in X_train])
        x_test_tf = tfq.convert_to_tensor([self.to_circuit(x) for x in X_test])

        model = self.build_model()
        # Early stopping if no improvement in 5 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        
        model.fit(x_train_tf, y_train, epochs=epochs, batch_size=8, 
                  validation_data=(x_test_tf, y_test), callbacks=[early_stop], verbose=1)
        
        _, acc = model.evaluate(x_test_tf, y_test, verbose=0)
        return acc
