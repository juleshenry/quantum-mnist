v have a clear understanding of the current implementation and have developed a four-phase plan to upgrade to a 5x5 quantum model with PCA-enhanced feature
  extraction. I will now examine phasefive/classical_k_classifier.py to ensure it can be properly aligned with the new 25-input requirement.

╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✓  ReadFile phasefive/classical_k_classifier.py                                                                                                               │
│                                                                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
✦ This plan outlines the steps to upscale the input resolution to 5x5 (25 qubits) and replace raw downsampling with PCA-based dimensionality reduction. This approach
  aims to capture more informative features from high-resolution plankton images while addressing the "Information Bottleneck."

  Proposed Refactor Plan: 5x5 PCA-Enhanced Quantum Pipeline


  1. Data Pipeline Refactor (phasefive/data_loader.py)
   * High-Res Loading: Update load_plankton_k_categories to load images at (32, 32) or higher (e.g., the original 128x128 if resources allow) to provide a rich
     feature set for PCA.
   * PCA Integration: Implement apply_pca_reduction(X_train, X_test, n_components=25):
       * Fit PCA on the training set and transform both training and testing sets.
       * Include a normalization step (Min-Max Scaling to [0, 1]) to ensure PCA components are mapped correctly to quantum rotation gates ($\pi \times
         \text{value}$).


  2. Quantum Architecture Update (phasefive/quantum_k_classifier.py)
   * Grid Expansion: Scale the qubit grid from cirq.GridQubit.rect(4, 4) to 5, 5.
   * Circuit Logic: Update convert_to_circuit to iterate over 25 flattened PCA components instead of 16 raw pixels.
   * Interaction Scaling: Update entanglement and parametric layers (XX, ZZ gates) to handle the 25 data qubits plus the readout qubit.


  3. Classical Model Realignment (phasefive/classical_k_classifier.py)
   * Input Shape: Adjust create_fair_classical_k_model to accept 25 inputs (input_shape=(25,)).
   * Fairness Calibration: Recalculate parameter counts to maintain parity with the expanded PQC (now ~50 parameters per layer).


  4. Experiment Orchestration (phasefive/run_experiments.py & phasefive/scientific_comparison.py)
   * Pipeline Logic: Modify the experiment scripts to insert the PCA step after loading but before model initialization.
   * Uniform Comparison: Ensure both the "Fair Classical" and "Quantum" models use the same 25 PCA features, isolating the "Quantum Advantage" check from the
     "Compression Quality" variable.


  5. Validation Strategy
   * Circuit Verification: Test that PCA features correctly drive the Ry gates and that the PQC compiles on the new 5x5 grid.
   * Comparison Check: Run a 2-category smoke test to verify the end-to-end pipeline before running full experiments.


