# quantum-mnist
This project seeks to confirm results published in https://thesai.org/Downloads/Volume11No10/Paper_40-Handwritten_Numeric_Image_Classification.pdf

We will explore quantum image processing via plankton data set found in this paper: https://arxiv.org/pdf/2108.05258.pdf

Other relevant research:
https://arxiv.org/pdf/2011.02831.pdf

### Project Phases
1. **confirm ipynb**: Confirm research conclusions in google colab using the MNIST dataset.
2. **basic binary quantum**: Apply a basic binary quantum classifier to the plankton dataset and compare with a "fair" classical neural net.
3. **optimise via param sweep**: Optimize the binary classifier through a hyperparameter sweep on neural architectures.
4. **compare to classical**: Perform the generalized quantum algorithm on the plankton dataset and compare results to established classical deep learning approaches.

---

# Phase 1: confirm ipynb
Done. We have tested the quantum mnist colab and confirmed it works as described in the original research.

# Phase 2: basic binary quantum
Done. We have implemented an improved binary quantum classifier (`phasetwo/binary_quantum_classifier.py`) using **Angle Encoding** and an expressive PQC with entanglement. This model consistently achieves >60% accuracy on multiple plankton pairs, significantly outperforming the initial threshold baseline.

### 1. Quantum Architecture: Expressive PQC with Angle Encoding
The model utilizes a hybrid classical-quantum pipeline. The classical layer prepares the morphological features, which are then injected into a high-expressivity quantum circuit.

```mermaid
graph TD
    %% Classical Layer
    subgraph Classical_Layer [STRATEGY 1: CLASSICAL PREPROCESSING]
        direction LR
        A["<br/><b>16x16 Grayscale</b><br/><b>Plankton Image</b><br/><br/>"] --> B["<br/><b>Downsample</b><br/><b>to 4x4</b><br/><br/>"]
        B --> C["<br/><b>Min-Max</b><br/><b>Normalization</b><br/><br/>"]
        C --> D["<br/><b>Feature Vector</b><br/><b>(16 dimensions)</b><br/><br/>"]
    end

    %% Data Handover
    D ==> Interface{{"<br/><b>Classical-Quantum Handover</b><br/><b>(θ = π · x)</b><br/><br/>"}}

    %% Quantum Layer
    subgraph Quantum_Layer [STRATEGY 2: QUANTUM CIRCUIT - PHASE 2]
        direction LR
        
        subgraph Register_Init [Register Init]
            direction TB
            Q_Data["<br/><b>Data Qubits</b><br/><b>|00...0⟩₁₆</b><br/><br/>"]
            Q_Anc["<br/><b>Ancilla Qubit</b><br/><b>|0⟩</b><br/><br/>"]
        end

        subgraph PQC_Flow [Quantum Processing Unit]
            direction LR
            E["<br/><b>Angle Encoding</b><br/><b>(Ry Gates)</b><br/><br/>"] --> F["<br/><b>Entanglement</b><br/><b>(CZ Chain)</b><br/><br/>"]
            F --> G["<br/><b>Parameterized</b><br/><b>Interactions</b><br/><b>(XX, ZZ, YY)</b><br/><br/>"]
        end
        
        Register_Init ==> PQC_Flow
        PQC_Flow --> H["<br/><b>Interference</b><br/><b>(Hadamard)</b><br/><br/>"]
        H --> I["<br/><b>Measurement</b><br/><b>(⟨Z⟩ Expectation)</b><br/><br/>"]
    end

    %% Final Output
    I ==> J["<br/><b>Binary Classification</b><br/><b>Result</b><br/><br/>"]

    %% Styling with High Contrast
    style Classical_Layer fill:#f0f0f0,stroke:#000,stroke-width:2px,color:#000
    style Quantum_Layer fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    style Interface fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style PQC_Flow fill:#fff,stroke:#01579b,stroke-dasharray: 5 5,color:#000
    style J fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#000
    
    %% Node-specific high contrast
    style A color:#000
    style B color:#000
    style C color:#000
    style D color:#000
    style E color:#000
    style F color:#000
    style G color:#000
    style H color:#000
    style I color:#000
    style Q_Data color:#000
    style Q_Anc color:#000
```

*   **Data Encoding (Angle Encoding):** Instead of binary thresholding ($x > 0.5$), we now use Angle Encoding. Each pixel $x_i$ from the downsampled 4x4 image is mapped to a rotation gate: $Ry(\pi \cdot x_i)$. This preserves the grayscale intensity information within the quantum state.
*   **Entanglement Layer:** Before interacting with the readout qubit, we introduce a linear chain of CZ (Controlled-Z) gates across all 16 data qubits. This allows the model to learn spatial correlations between pixels.
*   **Interaction Layers (XX, ZZ, YY):** The Parameterized Quantum Circuit (PQC) now uses three types of non-commuting interactions with the readout qubit:
    *   $XX$ interactions for bit-flip correlations.
    *   $ZZ$ interactions for phase-flip correlations.
    *   **New:** $YY$ interactions to increase the expressivity of the Hilbert space coverage.
*   **Readout:** A single ancilla qubit is initialized in the $|-\rangle$ state, undergoes the PQC interactions, and is measured in the Z-basis to produce the classification logit.

# Phase 3: optimise via param sweep
Done. We have transitioned the hyperparameter sweep (`phasethree/optimize_binary_classifier.py`) to the quantum domain, exploring encoding strategies, circuit depth, and optimization parameters. This allowed us to architect a quantum model that achieves >60% accuracy for at least 5 pairs of plankton.

### 2. Learned Hyperparameters
Through the sweep in `phasethree/optimize_binary_classifier.py`, the following configuration was identified as the most robust for complex plankton classification:

| Parameter | Optimal Value | Rationale |
| :--- | :--- | :--- |
| **Encoding** | `Angle` | Preserves grayscale features lost in Basis encoding. |
| **PQC Layers** | `1` | Higher layers (2+) showed signs of over-parameterization/noise on small 4x4 data. |
| **Learning Rate** | `0.01` | Needed for faster convergence with the Hinge loss function. |
| **Batch Size** | `16` | Provided better stochastic gradients for the quantum simulator. |
| **Loss Function** | `Hinge` | Maps naturally to the $[-1, 1]$ expectation value of the Z-measurement. |

### 3. Accuracy Results & Comparison
The model now consistently exceeds the 60% threshold. Below are the verified accuracies for the top performing pairs:

| Plankton Pair | Accuracy | Status |
| :--- | :--- | :--- |
| **aphanizomenon vs bosmina** | **62.7%** | Target Reached |
| **brachionus vs ceratium** | **73.9%** | Target Reached |
| **chaoborus vs conochilus** | **92.1%** | Target Reached |
| **copepod_skins vs cyclops** | **86.6%** | Target Reached |
| **daphnia vs daphnia_skins** | **72.9%** | Target Reached |
| **diaphanosoma vs diatom_chain** | **95.3%** | Target Reached |

# Phase 4: compare to classical
In this phase, we perform the generalized quantum algorithm on the plankton dataset and compare its performance to established classical deep learning approaches. 

We evaluate three levels of classical models (MobileNetV2, Custom CNN, and a "Fair" MLP) against our expressive Quantum Neural Network to determine the presence of a quantum advantage in parameter efficiency.

**Detailed Documentation:** [Phase 4 Neural Architectures & Comparison](phasefour/README.md)

### 1. Hybrid Comparison Pipeline
The phase 4 implementation (`phasefour/run_experiments.py`) compares models across different input resolutions and parameter scales:

```mermaid
graph TD
    A["Plankton Dataset"] --> B1["128x128 RGB"]
    A --> B2["4x4 Grayscale"]
    
    B1 --> C1["MobileNetV2 (Transfer)"]
    B1 --> C2["SmallCNN (Custom)"]
    
    B2 --> D1["Fair Classical MLP"]
    B2 --> D2["Quantum Neural Net (QNN)"]
    
    C1 & C2 & D1 & D2 --> E["Performance Comparison Matrix"]
```

*   **Classical SOTA:** Uses MobileNetV2 and a custom 4-layer CNN to establish a high-accuracy baseline on full-resolution data.
*   **Quantum QNN:** An expressive PQC using multi-axis (XX, ZZ, YY) interactions and entanglement.
*   **Fair Comparison:** A 35-parameter classical MLP used to benchmark the QNN's parameter efficiency on 4x4 data.

---

### Plankton Comparison Gallery
Visual examples of the morphological differences the model successfully distinguishes:

| Class A | Class B | Distinction |
| :--- | :--- | :--- |
| **Aphanizomenon** (Filamentous) | **Bosmina** (Water Flea) | Linear strands vs. rounded bodies. |
| <img src="data/zooplankton_0p5x/aphanizomenon/training_data/SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg" width="80"> | <img src="data/zooplankton_0p5x/bosmina/training_data/SPC-EAWAG-0P5X-1573585145749175-6767076553624-002309-001-1400-1412-108-100.jpeg" width="80"> | *Linear vs. Circular* |
| **Chaoborus** (Phantom Midge) | **Conochilus** (Rotifer Colony) | Elongated larvae vs. radial colonies. |
| <img src="data/zooplankton_0p5x/chaoborus/training_data/SPC-EAWAG-0P5X-1591718449551322-12463824312627-000419-026-1188-1246-84-352.jpeg" width="80"> | <img src="data/zooplankton_0p5x/conochilus/training_data/SPC-EAWAG-0P5X-1590768396346914-11513785197825-003869-026-2184-754-88-100.jpeg" width="80"> | *Elongated vs. Radial* |

---

## Plankton Gallery

A diverse 5x5 grid showing unique samples from 25 different plankton classes:

<table style="width: 100%; border-collapse: collapse; max-width: 600px;">
  <tr style="border: none;">
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/aphanizomenon/training_data/SPC-EAWAG-0P5X-1570543374882008-3725352526408-001649-138-1826-1500-72-56.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">aphanizomenon</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/asplanchna/training_data/SPC-EAWAG-0P5X-1526950998886503-1092852412937-037889-032-1750-512-140-152.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">asplanchna</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/asterionella/training_data/SPC-EAWAG-0P5X-1559498410191177-6403834470952-000009-178-1968-1974-44-44.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">asterionella</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/bosmina/training_data/SPC-EAWAG-0P5X-1538179481576808-3862027162424-002719-158-1584-732-108-92.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">bosmina</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/brachionus/training_data/SPC-EAWAG-0P5X-1536025824161581-1708400512198-066139-052-2704-1196-48-40.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">brachionus</span></td>
  </tr>
  <tr style="border: none;">
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1526947889580643-1089743154478-006799-026-1556-740-28-68.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">ceratium</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/chaoborus/training_data/SPC-EAWAG-0P5X-1562231004412805-9136388548324-001949-117-1452-1022-268-588.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">chaoborus</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/conochilus/training_data/SPC-EAWAG-0P5X-1542588436356357-8270916420626-028269-101-2428-534-176-176.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">conochilus</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/copepod_skins/training_data/SPC-EAWAG-0P5X-1556787841132187-3693306045404-002309-001-2208-2398-144-140.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">copepod_skins</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/cyclops/training_data/SPC-EAWAG-0P5X-1526948874674178-1090728236363-016649-023-1914-64-124-60.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">cyclops</span></td>
  </tr>
  <tr style="border: none;">
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/daphnia/training_data/SPC-EAWAG-0P5X-1526947603555156-1089457130702-003939-002-3054-1764-384-412.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">daphnia</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/daphnia_skins/training_data/SPC-EAWAG-0P5X-1563876012798986-10781374001834-000029-071-1438-998-92-168.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">daphnia_skins</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/diaphanosoma/training_data/SPC-EAWAG-0P5X-1529154310683986-868304612580-039009-103-1762-532-352-324.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">diaphanosoma</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/diatom_chain/training_data/SPC-EAWAG-0P5X-1580983315520363-1728849606663-001059-022-2288-1398-68-40.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">diatom_chain</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/dinobryon/training_data/SPC-EAWAG-0P5X-1527038777751010-1180630013044-051680-005-1148-1982-76-92.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">dinobryon</span></td>
  </tr>
  <tr style="border: none;">
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/dirt/training_data/SPC-EAWAG-0P5X-1555333517558724-2239004079652-003079-031-1310-1240-44-68.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">dirt</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/eudiaptomus/training_data/SPC-EAWAG-0P5X-1526948155607133-1090009176591-009459-025-1310-950-224-212.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">eudiaptomus</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/filament/training_data/SPC-EAWAG-0P5X-1526994879838086-1136732737852-044699-008-2524-1714-108-224.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">filament</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/fish/training_data/SPC-EAWAG-0P5X-1528679780230533-393780639499-045699-089-508-0-652-1084.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">fish</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/fragilaria/training_data/SPC-EAWAG-0P5X-1543666089762928-9348553504881-004799-014-3566-1624-48-28.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">fragilaria</span></td>
  </tr>
  <tr style="border: none;">
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539129903975531-4812435522148-002949-113-1646-0-512-684.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">hydra</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/kellicottia/training_data/SPC-EAWAG-0P5X-1526991690535448-1133543472742-012809-019-2850-724-64-64.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">kellicottia</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/keratella_cochlearis/training_data/SPC-EAWAG-0P5X-1555333647562566-2239134090459-004379-043-1018-734-28-60.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">keratella_cochlearis</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/keratella_quadrata/training_data/SPC-EAWAG-0P5X-1526948274619096-1090128186484-010649-031-2478-6-48-52.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">keratella_quadrata</span></td>
    <td align="center" style="border: none; padding: 2px;"><img src="./data/zooplankton_0p5x/leptodora/training_data/SPC-EAWAG-0P5X-1531181714690908-2895680804858-009049-245-1658-0-824-1112.jpeg" width="100" style="border-radius: 2px;" /><br/><span style="font-size: 8px;">leptodora</span></td>
  </tr>
</table>



---

## How to Run (Docker)

First, build the unified Docker image:

```bash
docker build -t quantum-mnist .
```

### Run Phase 2: Basic Binary Quantum (Data Ingress)
To verify the plankton data loading and class pairs:
```bash
docker run --rm quantum-mnist python phasetwo/plankton_ingress.py
```

### Run Phase 3: Optimise via Param Sweep
To see the hyperparameter sweep configuration:
```bash
docker run --rm quantum-mnist python phasethree/optimize_binary_classifier.py
```

### Run Phase 4: Compare to Classical (Full Experiments)
To run the full experiment suite and save results to your local machine:
```bash
docker run --rm -v $(pwd)/phasefour/results:/app/phasefour/results quantum-mnist python phasefour/run_experiments.py
```
