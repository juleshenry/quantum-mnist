# quantum-mnist
This project seeks to confirm results published in https://thesai.org/Downloads/Volume11No10/Paper_40-Handwritten_Numeric_Image_Classification.pdf

We will explore quantum image processing via plankton data set found in this paper: https://arxiv.org/pdf/2108.05258.pdf

Other relevant research:
https://arxiv.org/pdf/2011.02831.pdf

## Project Status

✅ **Phase 1**: Confirm research conclusions in google colab  
✅ **Phase 2**: Apply binary quantum classifier to plankton dataset  
✅ **Phase 3**: Generalize binary quantum classifier to N-class classification  
✅ **Phase 4**: Apply general quantum classifier to plankton dataset  

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the complete plankton quantum classification experiment:

```bash
python3 phasefour/plankton_quantum_algorithm.py
```

See individual phase README files for detailed documentation:
- [Phase Three: N-class Quantum Classifier](phasethree/README.md)
- [Phase Four: Plankton Classification](phasefour/README.md)

# Phase One: Confirm research conclusions in google colab
✅ **Done.** We have tested the quantum mnist colab and it works. The binary quantum classifier successfully distinguished between MNIST digits 3 and 6 using a 4x4 quantum circuit.

# Phase Two: Apply to plankton dataset (binary classification)
✅ **Done.** Applied the binary quantum classifier to the plankton dataset. Compared quantum approach with "fair" classical 4x4 neural net and full deep learning model.

Key findings:
- Converted plankton images to grayscale using Pillow
- Normalized and resized images for quantum processing
- Explored hyperparameters (batch_size, image size)
- Confirmed that 4x4 is limited but feasible for quantum simulation

# Phase Three: Generalize binary quantum classifier 
✅ **Done.** Implemented in `phasethree/quantum_classifier_n_class.py`

Generalized the binary classifier to support N-class classification using:
- **One-vs-rest approach**: Separate quantum circuit per class
- **Softmax output**: Multi-class probability distribution
- **Categorical cross-entropy loss**: Appropriate for multi-class problems
- **Tested on MNIST**: Validated on digits 0-9

See [Phase Three README](phasethree/README.md) for details.

# Phase Four: Apply general quantum classifier to plankton dataset
✅ **Done.** Implemented in `phasefour/plankton_quantum_algorithm.py`

Applied the generalized N-class quantum classifier to the plankton dataset (35 species) and compared with deep learning approaches as described in the [reference paper](https://arxiv.org/pdf/2108.05258.pdf).

**Implementation includes:**
- Multi-class quantum classifier for plankton species
- Classical CNN baseline (64x64 input)
- Fair classical model (4x4 input for direct comparison)
- Performance comparison and visualization

**Models compared:**
1. **Quantum Classifier** (4x4 input): Uses quantum circuits with PQC layers
2. **Fair Classical NN** (4x4 input): Simple feedforward network with same input constraints
3. **Deep Learning CNN** (64x64 input): Full convolutional network with larger input

See [Phase Four README](phasefour/README.md) for usage and details.

**Note:** Due to computational constraints of quantum simulation, the default configuration uses a subset of 5 plankton classes. For full 35-class classification, consider using quantum hardware or larger computational resources.
