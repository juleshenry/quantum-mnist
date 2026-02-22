import os
import sys
import sympy
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from PIL import Image
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PlanktonQuantumClassifier:
    def __init__(self, image_size=(4, 4)):
        self.sz = image_size
        self.n_q = image_size[0] * image_size[1]
        self.log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        # Define qubits once to ensure consistency
        self.qubits = cirq.GridQubit.rect(*self.sz)
        self.readout_qubit = cirq.GridQubit(-1, -1)
        print(f"Initialized Quantum Classifier with {self.n_q} qubits")

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def preprocess(self, path, augment=False):
        """Load, resize, and normalize images."""
        try:
            img = Image.open(path).convert("L").resize(self.sz, Image.BILINEAR)
            img_array = np.array(img) / 255.0
            
            # Contrast normalization
            if img_array.max() > img_array.min():
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            else:
                img_array = np.zeros_like(img_array)
            
            images = [img_array]
            if augment:
                images.append(np.flipud(img_array))
                images.append(np.fliplr(img_array))
                noisy = np.clip(img_array + np.random.normal(0, 0.05, img_array.shape), 0, 1)
                images.append(noisy)
            return images
        except Exception as e:
            self.log(f"Error preprocessing {path}: {e}")
            return [np.zeros(self.sz)]

    def to_circuit_improved(self, img):
        """Data Encoding: Encodes pixels into qubit rotations."""
        circuit = cirq.Circuit()
        flattened = img.flatten()
        for i, q in enumerate(self.qubits):
            # Map pixel intensity to a rotation angle
            angle = flattened[i] * np.pi
            circuit.append(cirq.ry(angle)(q))
        return circuit

    def build_model_improved(self, layers=1):
        """
        Quantum Model: Uses alphabetical symbol naming for compatibility 
        with older TFQ versions that lack 'model_circuit_symbols'.
        """
        circuit = cirq.Circuit()
        circuit.append(cirq.H(self.readout_qubit))

        for layer in range(layers):
            for i, q in enumerate(self.qubits):
                # Padding with :02d ensures q02 comes before q10 in alphabetical sorting
                name_x = f'alpha_layer{layer}_q{i:02d}_x'
                name_y = f'alpha_layer{layer}_q{i:02d}_y'
                circuit.append(cirq.rx(sympy.Symbol(name_x))(q))
                circuit.append(cirq.ry(sympy.Symbol(name_y))(q))
            
            # Entanglement Layer
            for i in range(self.n_q - 1):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
            
            # Measurement Connection
            circuit.append(cirq.CNOT(self.qubits[-1], self.readout_qubit))

        readout_op = cirq.Z(self.readout_qubit)

        # Build Keras Sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(
                circuit, 
                readout_op,
                differentiator=tfq.differentiators.ParameterShift()
            ),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_improved(self, cat_a, cat_b, data_dir, imgs_per=20, epochs=30):
        """Pipeline to load real data and train."""
        def get_files(category):
            path = os.path.join(data_dir, category)
            print(path)
            if not os.path.exists(path): return []
            return [os.path.join(path, f) for f in os.listdir(path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:imgs_per]

        self.log(f"Loading data for {cat_a} and {cat_b}...")
        files_a, files_b = get_files(cat_a), get_files(cat_b)
        
        if not files_a or not files_b:
            raise ValueError(f"Check data paths. Found A: {len(files_a)}, B: {len(files_b)}")

        x_raw, y = [], []
        for f in files_a:
            imgs = self.preprocess(f, augment=True)
            x_raw.extend(imgs)
            y.extend([0] * len(imgs))
        for f in files_b:
            imgs = self.preprocess(f, augment=True)
            x_raw.extend(imgs)
            y.extend([1] * len(imgs))

        y = np.array(y)
        x_train_raw, x_val_raw, y_train, y_val = train_test_split(
            x_raw, y, test_size=0.2, stratify=y, random_state=42
        )

        self.log("Converting to Quantum Tensors...")
        x_train = tfq.convert_to_tensor([self.to_circuit_improved(x) for x in x_train_raw])
        x_val = tfq.convert_to_tensor([self.to_circuit_improved(x) for x in x_val_raw])

        model = self.build_model_improved(layers=1)
        
        self.log("Starting training...")
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=4,
            validation_data=(x_val, y_val),
            verbose=1
        )
        return model, history

    def test_pipeline(self):
        """Synthetic test to ensure the environment is working."""
        self.log("\n--- Running Synthetic Test ---")
        x_synth, y_synth = [], []
        for i in range(60):
            img = np.zeros((4,4))
            if i < 30:
                img[:2, :] = 1.0; label = 0 # Top half bright
            else:
                img[2:, :] = 1.0; label = 1 # Bottom half bright
            x_synth.append(img)
            y_synth.append(label)
        
        x_tensor = tfq.convert_to_tensor([self.to_circuit_improved(x) for x in x_synth])
        y_tensor = np.array(y_synth)
        
        model = self.build_model_improved(layers=1)
        history = model.fit(x_tensor, y_tensor, epochs=15, verbose=0)
        
        acc = history.history['accuracy'][-1]
        self.log(f"Test Accuracy: {acc:.4f}")
        return acc > 0.65

def main():
    # 16 qubits (4x4) is computationally heavy for simulation. 
    # If it's too slow, change to (3,3) for 9 qubits.
    qc = PlanktonQuantumClassifier(image_size=(4, 4))
    
    if not qc.test_pipeline():
        print("Pipeline test failed. Continuing with real data...")

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/zooplankton_0p5x"
    
    if os.path.exists(data_path):
        categories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        if len(categories) >= 2:
            cat_a, cat_b = categories[0], categories[1]
            model, history = qc.train_improved(cat_a, cat_b, data_path)
            
            # Save using SavedModel format (creates a folder)
            model.save('quantum_plankton_model', save_format='tf')
            print("Done! Model saved to 'quantum_plankton_model' folder.")
        else:
            print(f"Error: Found only {len(categories)} categories in {data_path}. Need 2.")
    else:
        print(f"Data path '{data_path}' not found. Skipping real training.")

if __name__ == "__main__":
    main()