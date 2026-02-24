import os
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
    Optimized Quantum Binary Classifier (4x4).
    Reverting to 16 qubits for CPU stability.
    Uses 2 layers of XX/ZZ interactions and ParameterShift for reliability.
    """
    def __init__(self, image_size=(4, 4), threshold=0.5):
        self.image_size = image_size
        self.threshold = threshold
        self.data_qubits = cirq.GridQubit.rect(*self.image_size)
        self.readout_qubit = cirq.GridQubit(-1, -1)

    def preprocess_image(self, path, augment=True):
        try:
            img = Image.open(path).convert("L").resize(self.image_size, Image.BILINEAR)
            arr = np.array(img) / 255.0
            results = [arr]
            if augment:
                results.append(np.flipud(arr))
                results.append(np.fliplr(arr))
            return results
        except Exception as e:
            return []

    def remove_contradicting(self, xs, ys):
        mapping = collections.defaultdict(set)
        orig_x = {}
        for x, y in zip(xs, ys):
           orig_x[tuple(x.flatten())] = x
           mapping[tuple(x.flatten())].add(y)
        new_x, new_y = [], []
        for flatten_x in mapping:
          if len(mapping[flatten_x]) == 1:
              new_x.append(orig_x[flatten_x])
              new_y.append(next(iter(mapping[flatten_x])))
        return np.array(new_x), np.array(new_y)

    def convert_to_circuit(self, image):
        values = np.ndarray.flatten(image)
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            if value > self.threshold:
                circuit.append(cirq.X(self.data_qubits[i]))
        return circuit

    def build_model(self, n_layers=2):
        readout = self.readout_qubit
        data_qubits = self.data_qubits
        circuit = cirq.Circuit()
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))
        
        for l in range(n_layers):
            for i, qubit in enumerate(data_qubits):
                symbol_x = sympy.Symbol('xx-{}-{}'.format(l, i))
                circuit.append(cirq.XX(qubit, readout)**symbol_x)
                symbol_z = sympy.Symbol('zz-{}-{}'.format(l, i))
                circuit.append(cirq.ZZ(qubit, readout)**symbol_z)

        circuit.append(cirq.H(readout))
        readout_op = cirq.Z(readout)

        # Using ParameterShift differentiator for better stability on some CPUs
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(circuit, readout_op, differentiator=tfq.differentiators.ParameterShift()),
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

    def train(self, cat_a, cat_b, data_dir, imgs_per=100, epochs=30):
        def load_cat(category, label):
            path = os.path.join(data_dir, category, "training_data")
            if not os.path.exists(path): path = os.path.join(data_dir, category)
            files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpeg')][:imgs_per]
            imgs = []
            for f in files:
                imgs.extend(self.preprocess_image(f, augment=True))
            return imgs, [label] * len(imgs)

        imgs_a, labels_a = load_cat(cat_a, 1)
        imgs_b, labels_b = load_cat(cat_b, 0)
        X, y = self.remove_contradicting(np.array(imgs_a + imgs_b), np.array(labels_a + labels_b))
        tf_circs = tfq.convert_to_tensor([self.convert_to_circuit(x) for x in X])
        y_hinge = 2.0 * y - 1.0

        X_train, X_test, y_train, y_test = train_test_split(
            tf_circs.numpy(), y_hinge, test_size=0.2, stratify=y, random_state=42)
        
        X_train, X_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(X_test)
        model = self.build_model()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_hinge_accuracy', patience=5, mode='max', restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        return acc
