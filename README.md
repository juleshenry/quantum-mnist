# quantum-mnist
This project seeks to confirm results published in https://thesai.org/Downloads/Volume11No10/Paper_40-Handwritten_Numeric_Image_Classification.pdf

We will explore quantum image processing via plankton data set found in this paper: https://arxiv.org/pdf/2108.05258.pdfß

Other relevant research:
https://arxiv.org/pdf/2011.02831.pdf

-1. Confirm research conclusions in google colab
We have tested the quantum mnist colab and it works.
-2. Apply to the plankton data set. We will look at the cartesian product comparison of binary classification via the "fair" 4x4 neural net. Do the same for the quantum scenario.
We need to convert the plankton dataset to grayscale. We do so using the default configuration in Pillow. Next, we downsize colors to normalize. In the current iteration, the hyperparameters of batch_size and image size must be explored. The size of 4x4 is insufficient for the plankton dataset, so a larger quantum simulation will be needed than described in the 4x4 method. Even in 16x16, the naive FFN is not powerful. ✓ Done in Phase Two
-3. Generalize binary quantum classifier
In this phase, we devise a generalization that maps to images containing 0-9.
-4. Apply general quantum classifier to plankton dataset
In this phase, we perform the generalized algorithm on the plankton dataset and compare to the deep learning approach found here.

# Phase One: Confirm research conclusions in google colab
Done. We have tested the quantum mnist colab and it works. 

# Phase Two:
Apply lessons in Phase One to the plankton data set. We will look at the cartesian product comparison of binary classification via the "fair" 4x4 neural net. Do the same for the quantum scenario.

We need to convert the plankton dataset to grayscale. We do so using the default configuration in Pillow. Next, we downsize colors to normalize. In the current iteration, the hyperparameters of batch_size and image size must be explored. The size of 4x4 is insufficient for the plankton dataset, so a larger quantum simulation will be needed than described in the 4x4 method. Even in 16x16, the naive FFN is not powerful. ✓ Done in Phase Two

# Phase Three: Generalize binary quantum classifier 
In this phase, we devise a generalization that maps to images containing 0-9.

# Phase Four : Apply general quantum classifier to plankton dataset
In this phase, we perform the generalized algorithm on the plankton dataset and compare to the deep learning approach found [here](https://arxiv.org/pdf/2108.05258.pdf).

## Implementation Complete! ✓

We have successfully implemented a quantum neural network for plankton image classification:

### Key Features:
- **Quantum Architecture**: Hybrid quantum-classical model using TensorFlow Quantum and Cirq
- **Image Size**: Supports 8x8 (64 qubits) for practical simulation, scalable to 16x16
- **Encoding**: Binary threshold encoding of grayscale pixels to quantum states
- **Circuit Design**: Parameterized quantum circuit (PQC) with XX and ZZ entangling layers
- **Classification**: Binary classification between plankton species pairs

### Architecture Improvements over Phase One:
1. **Larger Quantum Circuits**: Scaled from 4x4 (16 qubits) to 8x8 (64 qubits)
2. **Enhanced Preprocessing**: Bilinear interpolation for better image quality
3. **Local Data Loading**: Direct loading from zooplankton dataset
4. **Flexible Training**: Configurable hyperparameters (batch size, epochs, image size)
5. **Cartesian Product Support**: Automated testing across multiple species pairs

### Files:
- `phasefour/plankton_quantum_algorithm.py` - Main implementation
- `phasefour/example_usage.py` - Usage examples and demonstrations
- `phasefour/README.md` - Detailed technical documentation

### Usage:
```python
from phasefour.plankton_quantum_algorithm import PlanktonQuantumClassifier

# Initialize with 8x8 images (64 qubits)
classifier = PlanktonQuantumClassifier(image_size=(8, 8))

# Train on binary classification task
model, history, accuracy = classifier.train_binary_classifier(
    category_a="bosmina",
    category_b="cyclops",
    plankton_dir="data/zooplankton_0p5x",
    max_images=30,
    epochs=15
)
```

See `phasefour/README.md` for complete documentation and technical details.
