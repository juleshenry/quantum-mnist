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
Apply lessons from Phase 1 to the plankton data set. We perform a cartesian product comparison of binary classification via a "fair" 16x16 classical neural net and its quantum equivalent.

We convert the plankton dataset to grayscale using the default configuration in Pillow and normalize pixel values. The 4x4 resolution is insufficient for the plankton dataset, so a larger 16x16 simulation is used. Even at 16x16, the naive FFN (with only 2 hidden neurons) remains underpowered, often achieving near-random accuracy on binary pairs.
(We are here)

# Phase 3: optimise via param sweep
In this phase, we set up a hyperparameter sweep on neural architectures. Note that k-class classification is likely impossible with such weak architectures, so we focus on optimizing the binary classification performance.

# Phase 4: compare to classical
In this phase, we perform the generalized quantum algorithm on the plankton dataset and compare its performance to the classical deep learning approach found [here](https://arxiv.org/pdf/2108.05258.pdf).

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
