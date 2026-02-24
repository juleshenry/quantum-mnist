import os
import sys
from src.classifiers.unified_classifier import UnifiedQuantumClassifier

def main():
    data_path = "data/zooplankton_0p5x"
    if not os.path.exists(data_path):
        print("Data path " + data_path + " not found.")
        return

    # Using bosmina vs filament as they should be highly distinct
    cat_a, cat_b = "bosmina", "filament"
    
    print("\n" + "="*60)
    print("RUNNING TUTORIAL-STYLE QUANTUM CLASSIFIER")
    print("Pair: {} vs {}".format(cat_a, cat_b))
    print("="*60)
    
    clf = UnifiedQuantumClassifier(image_size=(4, 4), threshold=0.5)
    acc = clf.train(cat_a, cat_b, data_path, imgs_per=100, epochs=30)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("Final Test Accuracy: {:.4f}".format(acc))
    print("="*60)
    
    if acc >= 0.60:
        print("\nSUCCESS: Accuracy is over 60%!")
    else:
        print("\nFAILED: Accuracy is below 60%.")

if __name__ == "__main__":
    main()