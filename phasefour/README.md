# Phase 4: Generalized Quantum vs. Classical Deep Learning

In this phase, we compare the generalized quantum algorithm's performance against established classical deep learning architectures. This includes a high-capacity CNN, transfer learning via MobileNetV2, and a "Fair Classical" model designed to match the parameter constraints of the quantum circuit.

## 1. Classical Neural Architectures

We evaluate three classical baselines to establish benchmarks across different scales of complexity.

### A. MobileNetV2 (Transfer Learning)
Leveraging a state-of-the-art architecture pretrained on ImageNet to identify complex morphological features in plankton.
- **Base**: MobileNetV2 (Frozen convolutional layers).
- **Head**: Custom classification head [Dropout(0.3) -> Linear(35)].
- **Input**: 128x128 RGB images.
- **Complexity**: High (Benefit of feature extraction from millions of images).

### B. SmallCNN (Custom)
A custom convolutional neural network designed specifically for this dataset.
- **Layers**: 4x [Conv2D -> ReLU -> MaxPool].
- **Channel Depth**: 32, 64, 128, 128.
- **Fully Connected**: 256 units with 50% Dropout.
- **Input**: 128x128 RGB images.
- **Complexity**: Medium (~2.3M parameters).

### C. "Fair" Classical Baseline
A minimal Multi-Layer Perceptron (MLP) that matches the low parameter count (~35) of the Quantum Neural Network.
- **Architecture**: Flatten -> Dense(2 units) -> ReLU -> Dense(1 unit).
- **Input**: 4x4 Grayscale images.
- **Purpose**: To provide a direct comparison of "Parameter Efficiency" between classical and quantum regimes.

---

## 2. Quantum Neural Architecture (QNN)

The Phase 4 QNN utilizes an expressive Parameterized Quantum Circuit (PQC) with multi-axis interactions, optimized for 4x4 downsampled features.

```mermaid
graph TD
    %% Classical Data Prep
    subgraph Classical_Prep [CLASSICAL PREPROCESSING]
        direction LR
        I["<br/><b>128x128 RGB</b><br/><b>Plankton Image</b><br/><br/>"] --> J["<br/><b>Grayscale &</b><br/><b>Resize (4x4)</b><br/><br/>"]
        J --> K["<br/><b>Min-Max</b><br/><b>Normalization</b><br/><br/>"]
        K --> L["<br/><b>Feature Vector</b><br/><b>(16 dims)</b><br/><br/>"]
    end

    %% Quantum Circuit
    subgraph Quantum_Circuit [QUANTUM CIRCUIT - PHASE 4]
        direction LR
        
        subgraph Encoding [Encoding Layer]
            direction TB
            E1["<br/><b>Angle Encoding</b><br/><b>Ry(π · x)</b><br/><br/>"]
        end

        subgraph Entanglement [Entanglement Layer]
            direction TB
            F1["<br/><b>CZ Chain</b><br/><b>(Linear)</b><br/><br/>"]
        end

        subgraph PQC_Interactions [Parameterized Interactions]
            direction LR
            G1["<br/><b>XX Gates</b><br/><br/>"] --> G2["<br/><b>ZZ Gates</b><br/><br/>"]
            G2 --> G3["<br/><b>YY Gates</b><br/><br/>"]
        end
    end

    %% Output
    L ==> Encoding
    Encoding ==> Entanglement
    Entanglement ==> PQC_Interactions
    PQC_Interactions --> M["<br/><b>Hadamard +</b><br/><b>Z-Readout</b><br/><br/>"]
    M ==> N["<br/><b>Classification</b><br/><b>Logit (Hinge)</b><br/><br/>"]

    %% Styling
    style Classical_Prep fill:#f5f5f5,stroke:#333,stroke-width:2px,color:#fff
    style Quantum_Circuit fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#fff
    style PQC_Interactions fill:#fff,stroke:#1565c0,stroke-dasharray: 5 5,color:#000
    style N fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    
    style I color:#fff
    style J color:#fff
    style K color:#fff
    style L color:#fff
    style E1 color:#fff
    style F1 color:#fff
    style G1 color:#fff
    style G2 color:#fff
    style G3 color:#fff
    style M color:#000
```

### Key Quantum Components:
*   **Angle Encoding:** Preserves 4-bit grayscale intensity by mapping pixel values to $Ry$ rotations.
*   **Linear Entanglement:** A chain of $CZ$ gates allows for spatial feature correlation across the 4x4 grid.
*   **Multi-Axis Interactions:** Uses parameterized $XX$, $ZZ$, and $YY$ interactions between data qubits and the readout qubit to capture non-commuting correlations.
*   **Optimization:** Trained using the **Hinge Loss** function, which is naturally suited for the $[-1, 1]$ expectation value of the quantum readout.

---

## 3. Comparison Metrics

| Model | Input Size | Params (Approx) | Data Format |
| :--- | :--- | :--- | :--- |
| **MobileNetV2** | 128x128 | ~45K (Head) | RGB |
| **SmallCNN** | 128x128 | ~2.3M | RGB |
| **Fair Classical** | 4x4 | 35 | Grayscale |
| **QNN** | 4x4 | 48 | Quantum (Angle) |
