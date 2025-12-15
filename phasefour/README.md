# Phase Four: Apply Quantum Classifier to Plankton Dataset

This phase implements a generalized quantum classifier for the plankton dataset and compares it with classical deep learning approaches.

## Overview

Building on the binary quantum classifier from Phase One and the multi-class classifier from Phase Three, this implementation:

1. Applies the quantum classification algorithm to the plankton dataset (35 classes)
2. Implements a classical CNN baseline for comparison (as described in the [reference paper](https://arxiv.org/pdf/2108.05258.pdf))
3. Provides a fair comparison model using the same 4x4 input as the quantum model
4. Generates performance metrics and comparison visualizations

## Files

- `plankton_quantum_algorithm.py`: Main implementation applying quantum classifier to plankton data

## Usage

### Basic Usage

Run the complete experiment with a subset of plankton classes:

```bash
python3 phasefour/plankton_quantum_algorithm.py
```

This will:
1. Load plankton images from the dataset
2. Train a quantum classifier (4x4 input)
3. Train a fair classical model (4x4 input)
4. Train a full CNN model (64x64 input)
5. Compare results and generate a comparison plot

### Configuration

You can modify the following parameters in `plankton_quantum_algorithm.py`:

- `SUBSET_CLASSES`: List of plankton classes to use (default: 5 classes for feasibility)
- `MAX_IMAGES_PER_CLASS`: Maximum images per class (default: 50)
- `epochs`: Training epochs for each model
- `batch_size`: Batch size for training

## Models Compared

### 1. Quantum Classifier (4x4)
- Uses quantum circuits to encode 4x4 binary images
- One-vs-rest approach with separate quantum circuits per class
- Based on TensorFlow Quantum's PQC (Parametrized Quantum Circuit) layers

### 2. Fair Classical Model (4x4)
- Simple feedforward neural network
- Same 4x4 input size as quantum model
- Provides fair comparison with similar input constraints

### 3. Deep Learning CNN (64x64)
- Full convolutional neural network
- Larger input size (64x64) for better feature extraction
- Based on approaches from the plankton classification literature

## Dependencies

Install required packages:

```bash
pip install -r ../requirements.txt
```

Key dependencies:
- TensorFlow 2.7.0
- TensorFlow Quantum 0.7.2
- Cirq 0.13.1
- NumPy
- Pillow
- Matplotlib

## Expected Results

The quantum classifier is expected to:
- Achieve comparable accuracy to the fair classical model on small input sizes
- Demonstrate quantum advantage in feature encoding for image data
- Provide insights for quantum machine learning on biological imaging data

The full CNN is expected to outperform both due to its larger input size and deeper architecture.

## Notes

- Quantum simulation is computationally expensive - training may take significant time
- Default configuration uses a subset of classes (5) for feasibility
- For full 35-class classification, consider using a quantum computer or larger computational resources
- Results are saved to `plankton_comparison.png`

## References

- [Plankton Classification Paper](https://arxiv.org/pdf/2108.05258.pdf)
- [Quantum MNIST Paper](https://thesai.org/Downloads/Volume11No10/Paper_40-Handwritten_Numeric_Image_Classification.pdf)
- [TensorFlow Quantum Documentation](https://www.tensorflow.org/quantum)
