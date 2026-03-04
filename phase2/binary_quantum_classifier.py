"""
  ★■╬▂▂▂▂▂◓□
 ☆◕◓◊◊▇▅◕⬤▽■●⬤
      ▽■◑▅▆◑★■╬◒.
          ⬤▂▄◔▽▽▅◒◕★
            ⬤▄▄○◈◔◓◊○⬤▼
             .▲◈█▆▅▇██▇▂■
                ☆☆■◑█▇███○
            ★★★★.  □█▲○██▄
           ◒▇╬◑●◊■□▂▲★███◐
           ■○╬╬■▇▄□▂ ☆◊██△
             ★▆█▅█○▂ ▼◈█▇◑
              ▲█★██▂  ☆██▆
              ▼█◐◐█▂  ▲██▆
              ★█□☆▼▽  ▲██▄
              ★█◓     ▲██▄
              ▲█◒     ▽◈█▆◕★
             ◓▄○.      ▽▅██▲
            ☆◒▅   ▽□□□■.□▇█▇△.
            ◊○⬤   ⬤▅▄▄◊  □███▂△
            ◊◊▄           ◑███▄△
           ◓▇◒△           △▇███◒
           ◐█◒            ▽△◒██▂
           ◕█◒    ☆○◈◈◈     ◒██▂
           ◕█◒    ▲████★    ◒██▂
          △●○▼    ▲████★  . ◒██▅△
        ★□▂▅◕     ▲████★  . ◒███▆△▽
       ◓◈◊◕◑     ☆▲████▅◈   ◓█████▂◑★
      ◕◊▼  ◒      ▲█████▂   ◓▆██████◐☆
     ●▆▽△▄▇●      ▲████▇╬   □◊███████▂
     ▽█.◑██●     .▼███◈◑◒   △⬤███████⬤
      ▆☆◑██●      .▽▽▽       ■███▄███▼
      ◔●◐▲▄●          .      ■██▇▼╬▄◐★
         ▆█●                 ▲╬██▂☆.
         ▆█●              .  ▼▇██▄
     .☆□●▇●◕                 ▽▆██▇◊■☆.
    ▽◑○⬤▼╬○△                 .□████▆█◑▲.
   ⬤◈★▽◔▅█╬▼                   ████▇███◐☆
 .⬤●▽△○█▇██○▲                 ■████▂████◒▽
 □▆.▼▂▅⬤╬██◕    ☆☆★★★★★☆★★.  .▂████◒◓████●
□◐▽☆◈○▲▂██◑●◓▼▲◓██████████╬◒◊███████◈□███▇◔
⬤●□█◊▼☆◐▆█◐◓██████████████████████▆●△☆◔███◒
⬤▆▄█⬤   ☆▂████◕◓◈◈◈◈◈◓■◈◈◈◈◒○███▆⬤▽    ▅██◒
▽◒◊◒★    .◔▅█▇▲             ◓█▇●▲      ⬤╬◐▼
           ☆▲★               ▼▼
                       /
   ___       ___  ___ (___       _ _
   |   )|   )|   )|   )|    |   )| | )
   |__/||__/ |__/||  / |__  |__/ |  /
       |

       /           /    /
   ___ (  ___  ___ (    (___  ___  ___
   |   )| |   )|   )|___)|    |   )|   )
   |__/ | |__/||  / | \  |__  |__/ |  /
   |

                   /    /
   _ _  ___  ___ (___    ___  ___
   | | )|   )|    |   )| |   )|___)
   |  / |__/||__  |  / | |  / |__


   /                     /
   (  ___  ___  ___  ___    ___  ___
   | |___)|   )|   )|   )| |   )|   )
   | |__  |__/||    |  / | |  / |__/
                               __/

               by Julian Henry
"""

# Note: This script requires tensorflow_quantum (TFQ) and cirq.
# It is designed to be run in a Google Colab environment or a local setup with TFQ.

# --- Phase 2: Viability of QNN on Binary Plankton Classification ---
# This script demonstrates the core feasibility of using a Parameterized
# Quantum Circuit (PQC) for real-world biological image identification.
# Uses stratified k-fold CV with seeding for reproducibility.

import json
import os
import cirq
import sympy
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

# import tensorflow_quantum as tfq  # Commented out to prevent local import errors

# Import local data loader
try:
    from phase2.plankton_ingress import load_images_for_class, get_plankton_names
except ImportError:
    # Fallback for colab if needed
    pass

# Import experiment utilities
try:
    from experiment_utils import set_seed, bootstrap_ci
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "experiment_utils",
        os.path.join(os.path.dirname(__file__), "..", "utils", "experiment_utils.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    set_seed = _mod.set_seed
    bootstrap_ci = _mod.bootstrap_ci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.01
LIMIT_PER_CLASS = 150
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class CircuitLayerBuilder:
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + "-" + str(i))
            circuit.append(gate(qubit, self.readout) ** symbol)


def convert_to_circuit(image):
    """Encode classical image into quantum datapoint using angle encoding."""
    # Input image is now already 4x4 from plankton_ingress
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        # Use Angle Encoding (Ry rotation)
        circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit


def create_quantum_model():
    """Create a QNN model circuit with improved expressivity."""
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()

    # Add entanglement between data qubits
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i + 1]))

    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")
    builder.add_layer(circuit, cirq.YY, "yy1")  # Added YY layer for more expressivity

    circuit.append(cirq.H(readout))
    return circuit, cirq.Z(readout)


def hinge_accuracy(y_true, y_pred):
    """Accuracy metric compatible with hinge-loss label format [-1, 1]."""
    y_true = tf.cast(y_true > 0.0, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def run_quantum_classification(class_a, class_b):
    """Run stratified k-fold CV quantum classification.

    Returns a dict with per-fold accuracies, mean, std, and bootstrap CI.
    """
    import tensorflow_quantum as tfq

    set_seed(SEED)

    # Load full dataset (no split yet — CV handles it)
    imgs_a = load_images_for_class(class_a, LIMIT_PER_CLASS)
    imgs_b = load_images_for_class(class_b, LIMIT_PER_CLASS)

    labels_a = np.zeros(len(imgs_a))
    labels_b = np.ones(len(imgs_b))

    X = np.array(imgs_a + imgs_b)
    y = np.concatenate([labels_a, labels_b])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_accuracies = []
    fold_losses = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        set_seed(SEED + fold_idx)  # per-fold determinism

        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Convert labels to hinge loss format [-1, 1]
        y_train_hinge = 2.0 * y_train - 1.0
        y_test_hinge = 2.0 * y_test - 1.0

        # Convert images to circuits
        x_train_circ = [convert_to_circuit(x) for x in x_train]
        x_test_circ = [convert_to_circuit(x) for x in x_test]

        x_train_tfq = tfq.convert_to_tensor(x_train_circ)
        x_test_tfq = tfq.convert_to_tensor(x_test_circ)

        model_circuit, model_readout = create_quantum_model()

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(), dtype=tf.string),
                tfq.layers.PQC(model_circuit, model_readout),
            ]
        )

        model.compile(
            loss=tf.keras.losses.Hinge(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=[hinge_accuracy],
        )

        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} "
              f"(train={len(x_train)}, test={len(x_test)}) ---")

        model.fit(
            x_train_tfq,
            y_train_hinge,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
        )

        loss, acc = model.evaluate(x_test_tfq, y_test_hinge, verbose=0)
        fold_accuracies.append(float(acc))
        fold_losses.append(float(loss))
        print(f"Fold {fold_idx + 1} — Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Aggregate results
    acc_ci = bootstrap_ci(fold_accuracies, seed=SEED)
    results = {
        "class_a": class_a,
        "class_b": class_b,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "limit_per_class": LIMIT_PER_CLASS,
        "fold_accuracies": fold_accuracies,
        "fold_losses": fold_losses,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_loss": float(np.mean(fold_losses)),
        "bootstrap_ci": acc_ci,
    }
    return results


if __name__ == "__main__":
    print("--- Phase 2: Binary Quantum Classification (Stratified 5-fold CV) ---")
    plank = get_plankton_names()
    if len(plank) < 2:
        print("Not enough plankton classes.")
    else:
        class_a, class_b = plank[0], plank[3]  # aphanizomenon vs bosmina
        print(f"Running Quantum Classification for {class_a} vs {class_b}")
        try:
            results = run_quantum_classification(class_a, class_b)

            print(f"\n=== Phase 2 Results ===")
            print(f"Accuracy: {results['mean_accuracy']:.4f} "
                  f"+/- {results['std_accuracy']:.4f}")
            ci = results['bootstrap_ci']
            print(f"95% Bootstrap CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
            print(f"Per-fold accuracies: {results['fold_accuracies']}")

            # Persist results
            os.makedirs(RESULTS_DIR, exist_ok=True)
            results_path = os.path.join(RESULTS_DIR, "phase2_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_path}")

        except Exception as e:
            print(f"Error during quantum execution: {e}")
            print("Note: This requires a working tensorflow_quantum installation.")
