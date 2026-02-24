import os
import sys
from src.classifiers.unified_classifier import UnifiedQuantumClassifier

def main():
    data_path = "data/zooplankton_0p5x"
    if not os.path.exists(data_path):
        print("Data path " + data_path + " not found.")
        return

    # Use bosmina (small, round) and dirt (irregular)
    cat_a, cat_b = "bosmina", "dirt"
    
    # 1. Testing with 5x5 resolution (25 qubits)
    print("\n" + "="*50)
    print("TESTING: 5x5 Resolution (25 Qubits) + PCA")
    print("="*50)
    clf_5x5 = UnifiedQuantumClassifier(image_size=(5, 5), encoding='angle')
    acc_5x5 = clf_5x5.train(cat_a, cat_b, data_path, epochs=30)
    print("\n5x5 PCA Accuracy: {:.4f}".format(acc_5x5))

    # 2. Testing with 4x4 resolution (16 qubits) as baseline
    print("\n" + "="*50)
    print("TESTING: 4x4 Resolution (16 Qubits) + PCA")
    print("="*50)
    clf_4x4 = UnifiedQuantumClassifier(image_size=(4, 4), encoding='angle')
    acc_4x4 = clf_4x4.train(cat_a, cat_b, data_path, epochs=30)
    print("\n4x4 PCA Accuracy: {:.4f}".format(acc_4x4))

    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print("5x5 PCA Accuracy: {:.4f}".format(acc_5x5))
    print("4x4 PCA Accuracy: {:.4f}".format(acc_4x4))
    
    best_acc = max(acc_5x5, acc_4x4)
    if best_acc >= 0.60:
        print("\nSUCCESS: Best accuracy {:.4f} is over 60%!".format(best_acc))
    else:
        print("\nFAILED: Best accuracy {:.4f} is below 60%.".format(best_acc))

if __name__ == "__main__":
    main()