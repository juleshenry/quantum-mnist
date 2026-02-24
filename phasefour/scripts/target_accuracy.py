import os
import sys
import sympy
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from PIL import Image
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TargetQuantumClassifier:
    def __init__(self, image_size=(4, 4)):
        self.sz = image_size
        self.n_q = image_size[0] * image_size[1]
        self.qubits = cirq.GridQubit.rect(*self.sz)
        self.readout_qubit = cirq.GridQubit(-1, -1)
        print(f"Initialized Target Quantum Classifier with {self.n_q} qubits")

    def preprocess(self, path):
        try:
            img = Image.open(path).convert("L").resize(self.sz, Image.BILINEAR)
            img_array = np.array(img) / 255.0
            if img_array.max() > img_array.min():
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            # Augmentation
            return [img_array, np.flipud(img_array), np.fliplr(img_array)]
        except Exception as e:
            return []

    def to_circuit(self, img):
        circuit = cirq.Circuit()
        flattened = img.flatten()
        for i, q in enumerate(self.qubits):
            # Angle encoding
            angle = flattened[i] * np.pi
            circuit.append(cirq.ry(angle)(q))
        return circuit

    def build_model(self, layers=2):
        circuit = cirq.Circuit()
        circuit.append(cirq.H(self.readout_qubit))

        for layer in range(layers):
            for i, q in enumerate(self.qubits):
                # Symbolic parameters
                circuit.append(cirq.rx(sympy.Symbol(f'x_{layer}_{i}'))(q))
                circuit.append(cirq.ry(sympy.Symbol(f'y_{layer}_{i}'))(q))
            
            # Entanglement Layer
            for i in range(self.n_q - 1):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
            
            # Connect to readout
            circuit.append(cirq.CNOT(self.qubits[-1], self.readout_qubit))

        readout_op = cirq.Z(self.readout_qubit)

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

def main():
    data_dir = "data/zooplankton_0p5x"
    # Choose pair: bosmina vs dirt
    cat_a, cat_b = "bosmina", "dirt"
    
    qc = TargetQuantumClassifier(image_size=(4, 4))
    
    def get_images(category):
        path = os.path.join(data_dir, category, "training_data")
        if not os.path.exists(path): return []
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpeg')][:40]
        imgs = []
        for f in files:
            imgs.extend(qc.preprocess(f))
        return imgs

    print(f"Loading {cat_a} and {cat_b}...")
    x_a = get_images(cat_a)
    x_b = get_images(cat_b)
    
    if len(x_a) == 0 or len(x_b) == 0:
        print(f"Error: No images found. Check data path: {data_dir}")
        sys.exit(1)

    X = x_a + x_b
    y = np.array([0]*len(x_a) + [1]*len(x_b))
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Dataset Size: {len(X_train_raw)} train, {len(X_test_raw)} test")
    print("Encoding...")
    x_train = tfq.convert_to_tensor([qc.to_circuit(x) for x in X_train_raw])
    x_test = tfq.convert_to_tensor([qc.to_circuit(x) for x in X_test_raw])

    model = qc.build_model(layers=1) # Start with 1 layer to avoid too many parameters
    
    print("Training...")
    history = model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=8,
        validation_data=(x_test, y_test),
        verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("\nFinal Test Accuracy: {:.4f}".format(acc))
    
    if acc >= 0.60:
        print("\nSUCCESS: Accuracy is over 60%!")
    else:
        print("\nFAILED: Accuracy is below 60%.")

if __name__ == "__main__":
    main()