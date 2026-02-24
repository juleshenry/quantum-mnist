import os
import sys
from src.classifiers.unified_classifier import UnifiedQuantumClassifier
from src.classifiers.simple_classifier import SimpleQuantumClassifier

def main():
    data_path = "data/zooplankton_0p5x"
    if not os.path.exists(data_path):
        print("Data path " + data_path + " not found.")
        return

    pairs = [
        ("bosmina", "filament"),
        ("cyclops", "daphnia"),
        ("asterionella", "brachionus")
    ]
    
    print("\n" + "="*60)
    print("COMPARATIVE BENCHMARK: UNIFIED (2-LAYER) VS SIMPLE (TUTORIAL)")
    print("Locked to 4x4 (16 qubits) for CPU safety")
    print("="*60)
    
    # 1. Run Unified (Optimized)
    print("\n>>> RUNNING UNIFIED CLASSIFIER (2 Layers, ParameterShift)")
    clf_unified = UnifiedQuantumClassifier(image_size=(4, 4))
    results_unified = []
    for cat_a, cat_b in pairs:
        try:
            acc = clf_unified.train(cat_a, cat_b, data_path, imgs_per=100, epochs=30)
            results_unified.append(acc)
        except Exception as e:
            print(f"Error in unified: {e}")
            results_unified.append(0.0)

    # 2. Run Simple (Tutorial)
    print("\n>>> RUNNING SIMPLE CLASSIFIER (Tutorial architecture)")
    clf_simple = SimpleQuantumClassifier(image_size=(4, 4))
    results_simple = []
    for cat_a, cat_b in pairs:
        try:
            acc = clf_simple.train(cat_a, cat_b, data_path, imgs_per=100, epochs=30)
            results_simple.append(acc)
        except Exception as e:
            print(f"Error in simple: {e}")
            results_simple.append(0.0)

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print("{:<30} | {:<10} | {:<10}".format("Pair", "Unified", "Simple"))
    print("-" * 60)
    for i, pair in enumerate(pairs):
        p_name = f"{pair[0]} vs {pair[1]}"
        print("{:<30} | {:<10.4f} | {:<10.4f}".format(p_name, results_unified[i], results_simple[i]))
    
    avg_uni = sum(results_unified) / len(results_unified)
    avg_sim = sum(results_simple) / len(results_simple)
    print("-" * 60)
    print("{:<30} | {:<10.4f} | {:<10.4f}".format("AVERAGE", avg_uni, avg_sim))

if __name__ == "__main__":
    main()