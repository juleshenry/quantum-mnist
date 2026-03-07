r"""
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
r"""

import os
import json
import numpy as np
import tensorflow as tf
import cirq
import sympy
import itertools
from sklearn.model_selection import StratifiedKFold

# Defer heavy / Docker-only imports
# import tensorflow_quantum as tfq

# Explicit module loading to avoid PYTHONPATH collision with phase4/data_loader
try:
    from phase2.plankton_ingress import (
        load_images_for_class, get_plankton_names, pca_transform,
    )
except ImportError:
    pass

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
SEED = 420
OUTER_FOLDS = 5          # outer CV for unbiased performance estimate
INNER_FOLDS = 3          # inner CV for hyperparameter selection
EPOCHS = 5               # epochs per training run (budget-constrained sweep)
LIMIT_PER_CLASS = 100
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Hyperparameter search space
HYPERPARAMS = {
    'n_layers': [1, 2],
    'learning_rate': [0.01, 0.001],
    'batch_size': [16, 32],
}


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def convert_to_circuit(pca_features):
    """Encode PCA-reduced features into quantum circuit using Ry angle encoding.

    Parameters
    ----------
    pca_features : array-like of shape (16,)
        PCA components scaled to [0, 1] via MinMaxScaler.
    """
    values = np.ndarray.flatten(pca_features)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit


def create_quantum_model(n_layers=1):
    """Create a parameterized quantum circuit with variable depth."""
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)

    for layer_idx in range(n_layers):
        builder.add_layer(circuit, cirq.XX, f"xx{layer_idx}")
        builder.add_layer(circuit, cirq.ZZ, f"zz{layer_idx}")

    return circuit, cirq.Z(readout)


def hinge_accuracy(y_true, y_pred):
    """Accuracy metric compatible with hinge-loss label format [-1, 1]."""
    y_true = tf.cast(y_true > 0.0, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def setup_sweep():
    """Generate all hyperparameter combinations."""
    keys, values = zip(*HYPERPARAMS.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations


# ---------------------------------------------------------------------------
# Core: Nested Cross-Validation
# ---------------------------------------------------------------------------

def _train_and_evaluate(tfq, config, x_train_pca, y_train, x_test_pca, y_test,
                        seed, model_cache):
    """Train a single model and return held-out accuracy.

    Parameters
    ----------
    x_train_pca, x_test_pca : ndarray of shape (N, 16)
        PCA-reduced features scaled to [0, 1].
    model_cache : dict
        Maps n_layers -> (model, initial_weights) to avoid rebuilding
        the tf.keras model (and triggering tf.function retracing).
    """
    set_seed(seed)

    x_train_tfq = tfq.convert_to_tensor(
        [convert_to_circuit(x) for x in x_train_pca]
    )
    x_test_tfq = tfq.convert_to_tensor(
        [convert_to_circuit(x) for x in x_test_pca]
    )

    n_layers = config['n_layers']
    if n_layers not in model_cache:
        model_circuit, model_readout = create_quantum_model(n_layers=n_layers)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(model_circuit, model_readout),
        ])
        model.compile(
            loss=tf.keras.losses.Hinge(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            metrics=[hinge_accuracy],
        )
        model_cache[n_layers] = (model, model.get_weights())

    model, initial_weights = model_cache[n_layers]

    # Reset weights + optimizer state for a clean training run.
    # We must NOT replace the optimizer object — assigning a new Adam()
    # would create fresh tf.Variables on the next model.fit(), which
    # crashes inside tf.function ("only supports singleton tf.Variables
    # created on the first call").  Instead, reset state in-place.
    model.set_weights(initial_weights)
    optimizer = model.optimizer
    optimizer.learning_rate.assign(config['learning_rate'])
    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))

    model.fit(
        x_train_tfq, 2.0 * y_train - 1.0,
        batch_size=config['batch_size'],
        epochs=EPOCHS,
        verbose=0,
    )
    _, acc = model.evaluate(
        x_test_tfq, 2.0 * y_test - 1.0, verbose=0
    )
    return float(acc)


def run_nested_cv(class_a, class_b):
    """Nested cross-validation for hyperparameter optimisation.

    Outer loop: ``OUTER_FOLDS``-fold stratified CV — each fold provides
    an unbiased accuracy estimate using the *best* hyper-parameters
    selected in the inner loop.

    Inner loop: ``INNER_FOLDS``-fold stratified CV on the outer-training
    set — selects the best configuration without touching the outer-test set.

    This eliminates the data leakage present in the original Phase 3
    implementation.
    """
    import tensorflow_quantum as tfq

    set_seed(SEED)

    combinations = setup_sweep()
    print(f"Hyperparameter combinations: {len(combinations)}")

    # Load full dataset
    imgs_a = load_images_for_class(class_a, LIMIT_PER_CLASS)
    imgs_b = load_images_for_class(class_b, LIMIT_PER_CLASS)
    labels_a = np.zeros(len(imgs_a))
    labels_b = np.ones(len(imgs_b))
    X = np.array(imgs_a + imgs_b)
    y = np.concatenate([labels_a, labels_b])

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED
    )

    outer_fold_results = []

    # Cache models by n_layers to avoid tf.function retracing
    model_cache = {}

    for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y)
    ):
        print(f"\n=== Outer Fold {outer_idx + 1}/{OUTER_FOLDS} ===")

        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

        # PCA on outer fold: fit on outer-train only (no leakage)
        X_outer_train_pca, X_outer_test_pca = pca_transform(
            X_outer_train, X_outer_test, seed=SEED + outer_idx
        )

        # --- Inner loop: select best config on outer-training data ---
        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True,
            random_state=SEED + outer_idx,
        )

        config_scores = {i: [] for i in range(len(combinations))}

        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(X_outer_train_pca, y_outer_train)
        ):
            X_inner_train = X_outer_train_pca[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            X_inner_val = X_outer_train_pca[inner_val_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            for cfg_idx, config in enumerate(combinations):
                inner_seed = SEED + outer_idx * 1000 + inner_idx * 100 + cfg_idx
                acc = _train_and_evaluate(
                    tfq, config,
                    X_inner_train, y_inner_train,
                    X_inner_val, y_inner_val,
                    seed=inner_seed,
                    model_cache=model_cache,
                )
                config_scores[cfg_idx].append(acc)
                print(
                    f"  Inner {inner_idx+1}/{INNER_FOLDS} | "
                    f"Config {cfg_idx+1}/{len(combinations)} | "
                    f"Val Acc: {acc:.4f}"
                )

        # Pick config with best mean inner-CV accuracy
        mean_inner = {
            i: float(np.mean(scores))
            for i, scores in config_scores.items()
        }
        best_cfg_idx = max(mean_inner, key=mean_inner.get)
        best_config = combinations[best_cfg_idx]
        best_inner_acc = mean_inner[best_cfg_idx]
        print(f"  Best inner config: {best_config} "
              f"(mean inner acc: {best_inner_acc:.4f})")

        # --- Retrain on full outer-training set, evaluate on outer-test ---
        outer_seed = SEED + outer_idx
        outer_acc = _train_and_evaluate(
            tfq, best_config,
            X_outer_train_pca, y_outer_train,
            X_outer_test_pca, y_outer_test,
            seed=outer_seed,
            model_cache=model_cache,
        )
        print(f"  Outer test accuracy: {outer_acc:.4f}")

        outer_fold_results.append({
            "outer_fold": outer_idx + 1,
            "best_config": best_config,
            "best_inner_mean_acc": best_inner_acc,
            "outer_test_accuracy": outer_acc,
        })

    # Aggregate
    outer_accs = [r["outer_test_accuracy"] for r in outer_fold_results]
    acc_ci = bootstrap_ci(outer_accs, seed=SEED)

    results = {
        "class_a": class_a,
        "class_b": class_b,
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "epochs_per_run": EPOCHS,
        "limit_per_class": LIMIT_PER_CLASS,
        "n_configs": len(combinations),
        "hyperparams": HYPERPARAMS,
        "per_fold": outer_fold_results,
        "outer_test_accuracies": outer_accs,
        "mean_accuracy": float(np.mean(outer_accs)),
        "std_accuracy": float(np.std(outer_accs)),
        "bootstrap_ci": acc_ci,
    }
    return results


if __name__ == "__main__":
    print("--- Phase 3: Hyperparameter Optimization (Nested CV) ---")
    print("Using nested cross-validation to prevent data leakage.\n")

    plank = get_plankton_names()
    if len(plank) < 2:
        print("Not enough plankton classes found.")
    else:
        class_a, class_b = plank[0], plank[3]  # aphanizomenon vs bosmina
        print(f"Optimizing Quantum Model for {class_a} vs {class_b}")

        results = run_nested_cv(class_a, class_b)

        print("\n=== Phase 3 Results (Nested CV) ===")
        print(f"Accuracy: {results['mean_accuracy']:.4f} "
              f"+/- {results['std_accuracy']:.4f}")
        ci = results['bootstrap_ci']
        print(f"95% Bootstrap CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"Per outer-fold accuracies: {results['outer_test_accuracies']}")
        for fold in results['per_fold']:
            print(f"  Fold {fold['outer_fold']}: "
                  f"acc={fold['outer_test_accuracy']:.4f}, "
                  f"config={fold['best_config']}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, "phase3_results.json")
        # Convert HYPERPARAMS list values for JSON serialization
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
