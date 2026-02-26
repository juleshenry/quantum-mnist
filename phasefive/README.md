# Phase Five: 5x5 PCA-Enhanced K-Category Quantum Classification

This phase extends the previous work to a multi-class ($k$-category) setting, upscaling the input resolution to a **5x5 grid (25 qubits)** and replacing raw downsampling with **PCA-based Dimensionality Reduction**.

## Methodology

### 1. The "Information Bottleneck" Rigor
- **The Problem:** Downsampling 128x128 images directly to 4x4 loses critical information (shape, texture).
- **The Solution:** We load images at 28x28 resolution and apply **Principal Component Analysis (PCA)** to extract the top 25 most informative features. These features are then scaled to $[0, 1]$ and encoded into 25 qubits.

### 2. Quantum Architecture (5x5 Grid)
- **Architecture:** 25 data qubits arranged in a 5x5 grid, plus 1 auxiliary readout qubit (**Total 26 qubits**).
- **Encoding:** Angle encoding (Ry rotations) of the 25 PCA features.
- **Circuit:** 1-2 layers of entangling CZ gates followed by parametric XX and ZZ gates between each data qubit and the readout qubit.
- **Parameter Count:** ~50 parameters per layer (25 XX + 25 ZZ), maintaining a compact, efficient model.

### 3. Scientific Comparison
- **Fair Classical:** A tiny MLP with 1-2 hidden units, recalibrated to match the ~50-100 parameters of the 25-qubit QNN.
- **CNN Baseline:** A standard CNN trained on full 28x28 resolution images to provide an "upper bound" on potential performance.
- **Scaling Study:** Performance is evaluated across $k \in \{2, 3, 4, 5, 8, 12, 16\}$ categories.

## Running the Experiments (Docker)

To run the standard multi-class scaling experiments:
```bash
docker build -t quantum-plankton .
docker run --rm -v $(pwd)/phasefive/results:/app/phasefive/results quantum-plankton python phasefive/run_experiments.py
```

To run the high-rigor swept comparison:
```bash
docker run -it --rm -v $(pwd)/phasefive/results:/app/phasefive/results quantum-plankton python phasefive/scientific_comparison.py
```

## Results
Results are saved in the `results/` directory:
- `comprehensive_k_results.csv`: Scaling statistics for all $k$ levels.
- `scientific_scaling_plot.png`: Comparison of swept models across categories.
