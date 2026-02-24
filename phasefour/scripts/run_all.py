"""
Master script to run all quantum binary classifiers.
This script executes the different implementations to verify the environment
and compare their basic functionality.
"""

import os
import sys
import subprocess

def run_script(script_path, args=[]):
    print("\n" + "="*80)
    print(f"RUNNING: {script_path} {' '.join(args)}")
    print("="*80 + "\n")
    
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n[SUCCESS] {script_path} finished successfully.")
    else:
        print(f"\n[FAILED] {script_path} exited with code {result.returncode}.")
    return result.returncode

def main():
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    sys.path.append(project_root)

    print("Quantum Binary Classifier Test Suite")
    print(f"Project Root: {project_root}")
    
    # 1. Run Simple Classifier (Synthetic data test)
    print("\nStep 1: Running Simple Classifier (Synthetic Test)")
    run_script("src/classifiers/simple_classifier.py")

    # 2. Run Example Usage script (includes preprocessing and small training test)
    print("\nStep 2: Running Example Usage Script")
    run_script("scripts/example_plankton_usage.py")

    # 3. Generate Comparison Report
    print("\nStep 3: Generating Comparison Report")
    run_script("src/utils/report_generator.py")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()