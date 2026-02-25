# Phase Five: K-Category Quantum Plankton Classification

This phase extends the previous binary classification work to a multi-class ($k$-category) setting, while strictly adhering to the 17-qubit architecture (16 data qubits + 1 readout qubit) used in Phase Three.

## Methodology

### 1. Quantum Algorithm
- **Architecture:** 16 data qubits arranged in a 4x4 grid, plus 1 auxiliary readout qubit.
- **Encoding:** Angle encoding on the 16 data qubits (downsampled 4x4 images).
- **Circuit:** 1 layer of entangling CZ gates followed by parametric XX and ZZ gates between each data qubit and the readout qubit.
- **Multi-class Strategy:** We utilize $k$ observables measured on the 17-qubit system. Specifically, we measure the expectation value of $Z$ on the readout qubit and the first $k-1$ data qubits. These $k$ values are then fed through a Softmax layer to produce class probabilities.
- **Parameter Count:** ~32 parameters for 1 layer.

### 2. Comparisons
- **Fair Classical:** A tiny MLP with 1 hidden unit, designed to have a parameter count (~27-33) nearly identical to the QNN.
- **CNN Baseline:** A standard CNN trained on higher-resolution (28x28) images.

### 3. Rigor & Scaling Study
- **Scaling Complexity:** We evaluate performance across a range of categories: $k \in \{2, 3, 5, 8, 16\}$.
- **Class Selection:** To ensure data quality, we automatically select the top $k$ most frequent plankton categories in the dataset.
- **Exposure Fairness:** To ensure a scientific comparison, the "Fair Classical" model is limited to the same number of training samples as the QNN, and its parameter count is scaled to remain as close as possible to the QNN's 32-parameter benchmark (e.g., ~33 parameters at $k=8$).
- **Statistics:** We report Mean and Standard Deviation for Accuracy and Macro-F1 scores across 3 independent trials for each $k$.

## Running the Experiment

To run the full scaling experiment using Docker:

```bash
docker build -t phasefive -f phasefive/Dockerfile .
docker run --rm -v $(pwd)/phasefive/results:/app/phasefive/results phasefive
```

## Results
Results are saved in the `results/` directory, including:
- `comprehensive_k_results.csv`: Per-trial statistics for all $k$ levels.
- `comprehensive_k_summary.csv`: Aggregated mean and standard deviation for scaling analysis.
- `k_scaling_comparison.png`: A scientific plot showing Accuracy and F1-Score trends as the number of categories increases.
