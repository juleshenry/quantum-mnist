import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_plankton_data
from classical_models import create_small_cnn, create_transfer_mobilenet

def train_and_evaluate():
    print("Loading data...")
    # Using 128x128 as per paper
    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_plankton_data(img_size=(128, 128))
    
    num_classes = len(classes)
    print(f"Loaded {len(X_train)} training images, {len(X_val)} validation, {len(X_test)} test.")
    print(f"Number of classes: {num_classes}")

    results = []

    # 1. Train Small CNN (similar to paper's conv4 but standardized)
    print("
Training Small CNN...")
    small_cnn = create_small_cnn(num_classes=num_classes)
    start_time = time.time()
    history_small = small_cnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    duration = time.time() - start_time
    test_loss, test_acc = small_cnn.evaluate(X_test, y_test, verbose=0)
    
    results.append({
        'model': 'SmallCNN',
        'test_accuracy': float(test_acc),
        'training_time': float(duration),
        'num_params': int(small_cnn.count_params())
    })
    print(f"SmallCNN Test Accuracy: {test_acc:.4f}")

    # 2. Train MobileNet (lightest model from paper)
    print("
Training MobileNetV2 (Transfer Learning)...")
    mobile_model = create_transfer_mobilenet(num_classes=num_classes)
    start_time = time.time()
    history_mobile = mobile_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    duration = time.time() - start_time
    test_loss_m, test_acc_m = mobile_model.evaluate(X_test, y_test, verbose=0)
    
    results.append({
        'model': 'MobileNetV2_Transfer',
        'test_accuracy': float(test_acc_m),
        'training_time': float(duration),
        'num_params': int(mobile_model.count_params())
    })
    print(f"MobileNetV2 Test Accuracy: {test_acc_m:.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/classical_phase_four_results.csv', index=False)
    
    with open('results/training_histories.json', 'w') as f:
        json.dump({
            'small_cnn': history_small.history,
            'mobilenet': history_mobile.history
        }, f)

    print("
Results saved to results/classical_phase_four_results.csv")

if __name__ == "__main__":
    train_and_evaluate()
