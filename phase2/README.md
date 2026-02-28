# Phase 2: Binary Quantum Classification - Data Processing

This phase focuses on adapting the quantum classification pipeline to handle real-world biological data (plankton images) which, unlike the MNIST dataset, comes in various shapes, sizes, and aspect ratios.

## Handling Image Variations

To account for the diverse morphological characteristics and varying dimensions of the plankton images, we implement a multi-stage normalization pipeline:

1.  **Grayscale Conversion:** All images are converted to single-channel grayscale (`'L'`). This reduces the input dimensionality while preserving the essential structural and texture features needed for classification.
2.  **Bilinear Interpolation:** We use a custom `bl_resize` (bilinear interpolation) algorithm to scale every image, regardless of its original size, to a fixed **16x16 resolution**.
    *   **Why Bilinear?** It provides a smoother transition between pixels compared to nearest-neighbor, which helps maintain edge information even when downsampling significantly.
3.  **Standardization vs. Distortion:** 
    *   By forcing all images into a 16x16 grid, we ensure the Quantum Neural Network (QNN) receives a consistent input vector size.
    *   **Note:** Because the scale factors for width and height are calculated independently, images with extreme aspect ratios (e.g., long filamentous *Aphanizomenon*) may appear "stretched" or "squashed." However, for this phase, the structural features preserved through interpolation are sufficient for the model to distinguish between classes.
4.  **Quantum Downsampling:** Before entering the quantum circuit, the images are further downsampled to **4x4** (16 pixels total). This is a computational necessity for current quantum simulators, as each pixel is encoded into a single qubit (16 qubits total).
5.  **Min-Max Normalization:** Pixel values are scaled to the range `[0, 1]`, which is then mapped to rotation angles $[0, \pi]$ for Angle Encoding in the quantum circuit.

## Key Files
- `plankton_ingress.py`: Handles the automated discovery of plankton classes and batch loading of images.
- `binary_quantum_classifier.py`: Implementation of the expressive PQC and the training loop.
- `Phase2.ipynb`: A comprehensive notebook for experimentation and visualization of the preprocessing steps.
