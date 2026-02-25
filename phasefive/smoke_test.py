import os
import numpy as np
import tensorflow as tf
from data_loader import load_plankton_k_categories
from quantum_k_classifier import create_qnn_multiclass_model, convert_to_circuit
import tensorflow_quantum as tfq

def smoke_test():
    print("Running Smoke Test...")
    CATEGORIES = ['aphanizomenon', 'bosmina', 'cyclops']
    K = len(CATEGORIES)
    
    # Load tiny data
    X, _, y, _ = load_plankton_k_categories(CATEGORIES, img_size=(4, 4))
    X = X[:10]
    y = y[:10]
    
    # Prepare circuits
    x_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X])
    
    # Create model
    model = create_qnn_multiclass_model(K)
    
    # Train for 1 epoch
    print("Testing training...")
    model.fit(x_circ, y, epochs=1, batch_size=2, verbose=1)
    
    # Predict
    print("Testing prediction...")
    preds = model.predict(x_circ)
    print(f"Predictions shape: {preds.shape}")
    assert preds.shape == (10, K)
    print("Smoke test passed!")

if __name__ == "__main__":
    try:
        smoke_test()
    except Exception as e:
        print(f"Smoke test failed: {e}")
