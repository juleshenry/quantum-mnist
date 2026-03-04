# Phase 5: PCA-Enhanced K-Category Quantum Classification (16 Qubits)

This phase extends the previous work to a multi-class ($k$-category) setting, encoding images into a **4x4 grid of 16 qubits** using **PCA-based dimensionality reduction** instead of raw downsampling.

## Methodology

### 1. The "Information Bottleneck" Solution
- **The Problem:** Downsampling 128x128 images directly to 4x4 loses critical morphological information (shape, texture, edge detail).
- **The Solution:** Images are loaded at 28x28 resolution (784 features) and **Principal Component Analysis (PCA)** extracts the top 16 most informative features. These features are whitened and scaled to $[0, 1]$ for quantum gate encoding.
- **Per-Fold PCA:** PCA is fit on each fold's training data and applied to the test data, preventing information leakage across folds.

### 2. Quantum Architecture (4x4 Grid)
- **Architecture:** 16 data qubits arranged in a 4x4 grid, plus 1 auxiliary readout qubit (**Total 17 qubits**).
- **Encoding:** Angle encoding ($R_y(\pi \cdot x)$ rotations) of the 16 PCA features.
- **Circuit:** 1-2 layers of entangling CZ gates followed by parametric XX and ZZ gates between each data qubit and the readout qubit.
- **Parameter Count:** ~32 parameters per layer (16 XX + 16 ZZ).

### 3. Scientific Comparison
- **Fair Classical:** A tiny MLP (`Dense(h) -> Dense(k)`) receiving the same 16 PCA features, calibrated to match the ~32-64 parameters of the QNN.
- **CNN Baseline:** A standard CNN trained on full 28x28 resolution images to provide an "upper bound" on potential performance.
- **Scaling Study:** Performance is evaluated across $k \in \{2, 3, 4, 5, 8, 12, 16\}$ categories.
- **Equal Data:** All models (QNN, Fair Classical, CNN) train on the same sample budget per fold.

### 4. PCA Pipeline Details
The PCA pipeline ensures a fair comparison by giving both QNN and Fair Classical the same information:
1. Load images at 28x28 grayscale (784 pixels)
2. Flatten to (N, 784) feature vectors
3. Fit PCA on training fold → 16 components (with whitening)
4. Transform both train and test → (N, 16)
5. MinMaxScaler to [0, 1] (fit on train, transform test)
6. QNN: encode via $R_y(\pi \cdot x_i)$ on 16 qubits
7. Fair Classical: feed directly as 16-dim input vector

This isolates the quantum-vs-classical comparison from the compression method.

## Running the Experiments (Docker)

To run the standard multi-class scaling experiments:
```bash
docker build --platform linux/amd64 -t quantum-plankton .
docker run --rm --platform linux/amd64 -v $(pwd)/phase5/results:/app/phase5/results quantum-plankton python phase5/run_experiments.py
```

To run the high-rigor swept comparison:
```bash
docker run --rm --platform linux/amd64 -v $(pwd)/phase5/results:/app/phase5/results quantum-plankton python phase5/scientific_comparison.py
```

Smoke test (quick verification of PCA pipeline):
```bash
docker run --rm --platform linux/amd64 quantum-plankton python phase5/smoke_test.py
```

## Results
Results are saved in the `results/` directory:
- `comprehensive_k_results.csv`: Per-fold metrics for all $k$ levels.
- `comprehensive_k_summary.csv`: Aggregated metrics with significance tests.
- `k_scaling_comparison.png`: Accuracy/F1 comparison plots.
- `experiment_config.json`: Full configuration including PCA parameters.

## 5. K-Category Scaling Benchmarks
We evaluate the QNN's ability to handle increasing classification complexity (k=2, 3, 4, 8, 12, 16). As the number of categories grows, the information bottleneck of the 16-qubit architecture becomes more apparent.

<!-- P5_RESULTS_START -->
| K (Categories) | QNN (PCA 16) | Fair Classical (PCA 16) |
| --- | --- | --- |
| *Results will be populated after running experiments with PCA pipeline* | | |
<!-- P5_RESULTS_END -->

*Note: Previous results used raw 4x4 downsampled pixels. The PCA-enhanced pipeline should yield improved feature quality and more meaningful quantum-classical comparisons.*
