# Quantum Image Classification: MNIST & Zooplankton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Quantum](https://img.shields.io/badge/TF%20Quantum-Latest-orange.svg)](https://www.tensorflow.org/quantum)

This research repository explores the application of **Quantum Neural Networks (QNNs)** to image classification tasks. It bridges two distinct research domains:
1. **Quantum Machine Learning (QML)**: Benchmarking against the foundational Quantum MNIST architectures.
2. **Freshwater Ecology**: Extending quantum classification to the **ZooLake dataset**, as introduced by [Kyathanahally et al. (2021)](https://arxiv.org/abs/2108.05258).

The project implements a hybrid quantum-classical pipeline that translates high-dimensional biological imagery into parameterized quantum circuits for binary and multi-class classification.

---

## 🔬 Research Context

### Research Progression
Although the repository is organized by functional directories, the research followed a four-phase progression:
1.  **Phase One (Baseline)**: Reproduction of Quantum MNIST architectures on handwritten digits (see `notebooks/01_mnist_baseline.ipynb`).
2.  **Phase Two (Extension)**: Initial application to the ZooLake dataset and architecture benchmarking.
3.  **Phase Three (Generalization)**: Development of a generalized N-class quantum framework.
4.  **Phase Four (Optimization)**: Final benchmarking of hybrid models and classical parity checks.

### Quantum Architecture Technical Pipeline (Phase Two)
The following diagram details the hybrid quantum-classical pipeline, contrasting the **Simple (Tutorial-based)** and **Unified (Variational)** strategies.

```mermaid
graph TD
    subgraph "Data Preprocessing"
        A[Raw Plankton Image] --> B[L-Mode Grayscale]
        B --> C[Bilinear Downsampling 4x4/8x8]
        C --> D[Min-Max Normalization]
        D --> E[Binary Threshold Encoding]
    end

    subgraph "Quantum State Preparation"
        E --> F["|ψ_in⟩ = ⨂_{i=1}^{N} |x_i⟩"]
        F --> G[N Qubits: 1 per Pixel]
    end

    subgraph "Circuit Architectures Compared"
        G --> H{Architecture Selection}
        
        subgraph "Simple (Farhi-style)"
            H --> S1[Readout Ancilla |+⟩]
            S1 --> S2[XX Gate Interaction Layers]
            S2 --> S3[ZZ Entangling Topology]
            S3 --> S4[Z-Basis Readout]
        end
        
        subgraph "Unified (Variational)"
            H --> V1[Parameterized Rotation Gates Ry/Rz]
            V1 --> V2[CNOT/CZ Entangling Layers]
            V2 --> V3[Multi-Layer Variational PQC]
            V3 --> V4[ParameterShift Differentiator]
        end
    end

    subgraph "Hybrid Classification"
        S4 --> J[Expectation Value ⟨Z⟩]
        V4 --> J
        J --> K[Classical Dense Layer]
        K --> L[Sigmoid Activation]
        L --> M[Binary Cross-Entropy Loss]
    end

    subgraph "Phase Two Benchmark Results"
        M --> N[Simple Accuracy: 64.06%]
        M --> O[Unified Accuracy: 52.69%]
    end

    style N fill:#f9f,stroke:#333,stroke-width:2px
    style O fill:#bbf,stroke:#333,stroke-width:2px
```

### Contributions
*   **Hybrid Quantum Pipeline**: Implementation of a full lifecycle from raw biological images to quantum circuit expectations.
*   **Comparative Benchmarking**: Direct comparison between "Simple" (Farhi-style) and "Unified" (Variational) quantum architectures.
*   **Classical Baselines**: Re-implementation of the 2021 ZooLake deep learning models (EfficientNet, DenseNet, etc.) for performance parity checks.

---

## 📁 Repository Structure

```text
.
├── data/                   # ZooLake dataset subset (zooplankton_0p5x)
├── docs/
│   ├── papers/             # Reference research papers
│   └── reports/            # Technical summaries and benchmark comparisons
├── notebooks/              # Step-by-step research phases
│   ├── 01_mnist_baseline.ipynb         # Reproduction of Quantum MNIST
│   ├── 02_plankton_extension.ipynb    # Extension to biological data
│   └── 03_plankton_experimentation.ipynb # Hyperparameter and architecture tuning
├── src/
│   ├── classifiers/        # Quantum Neural Network implementations
│   ├── classical_baselines/# PyTorch implementations of the 2021 paper
│   └── utils/              # Data ingress and quantum encoding utilities
├── scripts/                # Execution scripts for large-scale benchmarks
├── tests/                  # Validation and unit tests
├── Dockerfile              # Containerized environment for reproducibility
└── requirements.txt        # Project dependencies
```

---

## 🚀 Installation

### Prerequisites
*   Python 3.9 (Required for TensorFlow Quantum compatibility)
*   [Optional] Docker for containerized execution

### Local Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/quantum-mnist.git
cd quantum-mnist

# Create a virtual environment
python3.9 -m venv venv
source venv/bin/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage

### Running Quantum Classifiers
To execute the unified quantum classifier benchmark on the plankton dataset:
```bash
python scripts/run_unified_classifier.py
```

### Classical Baselines
To run the classical PyTorch implementation (Transfer Learning) on the same data:
```bash
python src/classical_baselines/kyathanahally_implementation.py
```

### Notebooks
For interactive exploration, start with `notebooks/01_mnist_baseline.ipynb` to understand the quantum encoding mechanism before moving to biological applications.

---

## ⚙️ Methodology

### 1. Quantum Encoding
We utilize **Binary Threshold Encoding**. Raw imagery is downsampled (typically to 4x4 or 8x8) and converted to grayscale. Pixel intensities are mapped to quantum states:
*   Black (0) $\rightarrow |0\rangle$
*   White (1) $\rightarrow |1\rangle$

### 2. Circuit Architecture
The repository supports multiple **Parameterized Quantum Circuit (PQC)** designs:
*   **Simple Classifier**: Single-layer interaction using XX gates followed by ZZ entanglement. High efficiency for low-resolution (4x4) inputs.
*   **Unified Classifier**: Multi-layer variational circuit designed for deeper feature extraction, utilizing the `ParameterShift` differentiator for hardware-ready gradients.

### 3. Classical Integration
The quantum circuit output (expectation values) is fed into a classical softmax or sigmoid layer, creating a **Hybrid Quantum-Classical Neural Network**.

---

## 📊 Key Findings
*   **Resolution Constraints**: Current quantum simulations are optimal at 4x4 and 8x8 resolutions. 16x16 simulations (256 qubits) require significant computational overhead.
*   **Accuracy**: The Simple QNN architecture achieved ~64% accuracy on distinct plankton species pairs at 4x4 resolution, demonstrating feasibility in near-term quantum classification.

---

## 📜 Citations

### Original Research
```bibtex
@article{kyathanahally2021deep,
  title={Deep learning models for classifying lake zooplankton and large phytoplankton colonies},
  author={Kyathanahally, S. P. and others},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```

### Implementation & Adaptation
```bibtex
@software{henry2026quantum,
  author = {Julian Henry},
  title = {Quantum MNIST & Plankton Classification: A Hybrid QNN Approach},
  institution = {aeae.inc},
  year = {2026},
  url = {https://github.com/aeae-inc/quantum-mnist}
}
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
