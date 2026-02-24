import os
import sys
import collections
import sympy
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from PIL import Image
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class UnifiedQuantumClassifier:
    """
    Unified Quantum Binary Classifier following the exact Farhi et al. approach
    from the TensorFlow Quantum MNIST tutorial.
    """
    
    def __init__(self, image_size=(4, 4), threshold=0.5):
        self.image_size = image_size
        self.n_qubits = image_size[0] * image_size[1]
        self.threshold = threshold
        
        # Define qubits
        self.data_qubits = cirq.GridQubit.rect(*self.image_size)
        self.readout_qubit = cirq.GridQubit(-1, -1)
        
        print("Initialized Tutorial-style Classifier: {}x{} images, {} qubits".format(
            image_size[0], image_size[1], self.n_qubits))

    def remove_contradicting(self, xs, ys):
        """Removes images that map to both labels (from the tutorial)."""
        mapping = collections.defaultdict(set)
        orig_x = {}
        for x, y in zip(xs, ys):
           orig_x[tuple(x.flatten())] = x
           mapping[tuple(x.flatten())].add(y)
        
        new_x = []
        new_y = []
        for flatten_x in mapping:
          x = orig_x[flatten_x]
          labels = mapping[flatten_x]
          if len(labels) == 1:
              new_x.append(x)
              new_y.append(next(iter(labels)))
        
        print("Unique images: {}, Contradictory removed: {}".format(
            len(mapping), len(xs) - len(new_x)))
        return np.array(new_x), np.array(new_y)

    def convert_to_circuit(self, image):
        """Binary encoding as per tutorial."""
        values = np.ndarray.flatten(image)
        qubits = cirq.GridQubit.rect(*self.image_size)
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            if value > self.threshold:
                circuit.append(cirq.X(qubits[i]))
        return circuit

    def build_model(self):
        """Build the specific XX/ZZ model from the tutorial."""
        readout = cirq.GridQubit(-1, -1)
        data_qubits = cirq.GridQubit.rect(*self.image_size)
        circuit = cirq.Circuit()
        
        # Prepare the readout qubit.
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))
        
        # XX Layers
        for i, qubit in enumerate(data_qubits):
            symbol = sympy.Symbol('xx-' + str(i))
            circuit.append(cirq.XX(qubit, readout)**symbol)
            
        # ZZ Layers
        for i, qubit in enumerate(data_qubits):
            symbol = sympy.Symbol('zz-' + str(i))
            circuit.append(cirq.ZZ(qubit, readout)**symbol)

        # Final readout preparation
        circuit.append(cirq.H(readout))
        readout_op = cirq.Z(readout)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(circuit, readout_op),
        ])
        
        def hinge_accuracy(y_true, y_pred):
            y_true = tf.squeeze(y_true) > 0.0
            y_pred = tf.squeeze(y_pred) > 0.0
            return tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))

        model.compile(
            loss=tf.keras.losses.Hinge(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[hinge_accuracy]
        )
        return model

    def train(self, cat_a, cat_b, data_dir, imgs_per=100, epochs=20):
        def load_cat(category, label):
            path = os.path.join(data_dir, category, "training_data")
            if not os.path.exists(path): path = os.path.join(data_dir, category)
            files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpeg')][:imgs_per]
            
            imgs = []
            for f in files:
                img = Image.open(f).convert("L").resize(self.image_size, Image.BILINEAR)
                imgs.append(np.array(img) / 255.0)
            return imgs, [label] * len(imgs)

        print("Loading {} vs {}...".format(cat_a, cat_b))
        imgs_a, labels_a = load_cat(cat_a, 1) # True
        imgs_b, labels_b = load_cat(cat_b, 0) # False
        
        X = np.array(imgs_a + imgs_b)
        y = np.array(labels_a + labels_b)

        # 1. Remove contradictory examples after downscaling
        X, y = self.remove_contradicting(X, y)

        # 2. Convert to circuits
        tf_circuits = tfq.convert_to_tensor([self.convert_to_circuit(x) for x in X])
        
        # 3. Hinge labels: [-1, 1]
        y_hinge = 2.0 * y - 1.0

        X_train, X_test, y_train, y_test = train_test_split(
            tf_circuits.numpy(), y_hinge, test_size=0.2, stratify=y_hinge, random_state=42)
        
        X_train = tf.convert_to_tensor(X_train)
        X_test = tf.convert_to_tensor(X_test)

        model = self.build_model()
        
        print("Starting training (Tutorial Style)...")
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        return acc
