import tensorflow as tf
import numpy as np
import itertools
from phasetwo.plankton_ingress import prepare_binary_dataset, get_plankton_names

# Configuration
QUBIT_DIMS = (16, 16)

def create_model(hidden_layers, neurons_per_layer, activation='relu', learning_rate=0.001):
    """
    Creates a classical neural network model based on hyperparameters.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(*QUBIT_DIMS, 1)))
    
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation))
        
    model.add(tf.keras.layers.Dense(1)) # Binary output (logits)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model

# Define the hyperparameter search space
hyperparams = {
    'hidden_layers': [1, 2, 3],
    'neurons_per_layer': [4, 8, 16],
    'activation': ['relu', 'tanh'],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [16, 32]
}

def setup_sweep():
    """
    Sets up the grid search by generating all combinations of hyperparameters.
    """
    keys, values = zip(*hyperparams.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to explore: {len(combinations)}")
    return combinations

def run_sweep(combinations, x_train, y_train, x_test, y_test):
    """
    Executes the hyperparameter sweep.
    """
    best_accuracy = 0
    best_config = None
    
    for i, config in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: {config}")
        model = create_model(
            hidden_layers=config['hidden_layers'],
            neurons_per_layer=config['neurons_per_layer'],
            activation=config['activation'],
            learning_rate=config['learning_rate']
        )
        
        # Expand dims for channel
        x_train_expanded = np.expand_dims(x_train, -1)
        x_test_expanded = np.expand_dims(x_test, -1)
        
        history = model.fit(
            x_train_expanded, y_train, 
            batch_size=config['batch_size'], 
            epochs=10, 
            verbose=0,
            validation_data=(x_test_expanded, y_test)
        )
        
        val_acc = max(history.history['val_accuracy'])
        print(f"Best Val Acc: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = config
            
    return best_config, best_accuracy

if __name__ == "__main__":
    plank = get_plankton_names()
    if len(plank) < 2:
        print("Not enough plankton classes found.")
    else:
        # Use a representative pair for optimization
        class_a, class_b = plank[0], plank[3] # aphanizomenon vs bosmina
        print(f"Optimizing for {class_a} vs {class_b}")
        
        (x_train, y_train), (x_test, y_test) = prepare_binary_dataset(class_a, class_b, limit=100)
        
        combos = setup_sweep()
        best_cfg, best_acc = run_sweep(combos, x_train, y_train, x_test, y_test)
        
        print("\n--- SWEEP COMPLETE ---")
        print(f"Best Configuration: {best_cfg}")
        print(f"Best Accuracy: {best_acc:.4f}")
