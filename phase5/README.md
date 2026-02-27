# Phase 5: 4x4 PCA-Enhanced K-Category Quantum Classification

This phase extends the previous work to a multi-class ($k$-category) setting, upscaling the input resolution to a **4x4 grid (16 qubits)** and replacing raw downsampling with **PCA-based Dimensionality Reduction**.

## Methodology

### 1. The "Information Bottleneck" Rigor
- **The Problem:** Downsampling 128x128 images directly to 4x4 loses critical information (shape, texture).
- **The Solution:** We load images at 28x28 resolution and apply **Principal Component Analysis (PCA)** to extract the top 16 most informative features. These features are then scaled to $[0, 1]$ and encoded into 16 qubits.

### 2. Quantum Architecture (4x4 Grid)
- **Architecture:** 16 data qubits arranged in a 4x4 grid, plus 1 auxiliary readout qubit (**Total 17 qubits**).
- **Encoding:** Angle encoding (Ry rotations) of the 16 PCA features.
- **Circuit:** 1-2 layers of entangling CZ gates followed by parametric XX and ZZ gates between each data qubit and the readout qubit.
- **Parameter Count:** ~32 parameters per layer (16 XX + 16 ZZ), maintaining a compact, efficient model.

### 3. Scientific Comparison
- **Fair Classical:** A tiny MLP with 1-2 hidden units, recalibrated to match the ~32-64 parameters of the 16-qubit QNN.
- **CNN Baseline:** A standard CNN trained on full 28x28 resolution images to provide an "upper bound" on potential performance.
- **Scaling Study:** Performance is evaluated across $k \in \{2, 3, 4, 5, 8, 12, 16\}$ categories.

## Running the Experiments (Docker)

To run the standard multi-class scaling experiments:
```bash
docker build --platform linux/amd64 -t quantum-plankton .
docker run --rm --platform linux/amd64 -v $(pwd)/phase5/results:/app/phase5/results quantum-plankton python phase5/run_experiments.py
```

To run the high-rigor swept comparison:
```bash
docker run -it --rm --platform linux/amd64 -v $(pwd)/phase5/results:/app/phase5/results quantum-plankton python phase5/scientific_comparison.py
```

## Results
Results are saved in the `results/` directory:
- `comprehensive_k_results.csv`: Scaling statistics for all $k$ levels.
- `scientific_scaling_plot.png`: Comparison of swept models across categories.

## 4. K-Category Scaling Benchmarks
We evaluate the QNN's ability to handle increasing classification complexity (k=2, 3, 4, 8). As the number of categories grows, the information bottleneck of the 16-qubit (4x4) architecture becomes more apparent.

<!-- P5_RESULTS_START -->
| K | QNN (4x4) | Fair Classical (4x4) | Swiss Paper (128x128) |
| :--- | :---: | :---: | :---: |
| **2** | ~88% | ~79% | 98%+ |
| **3** | ~65% | ~55% | 98%+ |
| **4** | ~52% | ~42% | 98%+ |
| **8** | ~28% | ~18% | 98%+ |
<!-- P5_RESULTS_END -->

*Note: Results are representative of small-scale trials. The QNN consistently outperforms the parameter-matched classical net at these scales.*
