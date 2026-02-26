# Phase Six: Quantum Interpretability (Saliency Maps)

This phase explores **Quantum Interpretability** by calculating gradients of the Quantum Neural Network (QNN) output with respect to the input features. These gradients are then projected back onto the original image space to create "Saliency Maps."

## Methodology

### 1. Differentiable QNN
- **The Challenge:** In previous phases, circuits were pre-computed and passed as serialized strings to the model. This broke the gradient flow between the input features and the output.
- **The Solution:** We implement a custom Keras model (`QuantumSaliencyModel`) that uses `tfq.layers.ControlledPQC`. The input PCA features are passed as parameter values for an angle-encoding circuit layer, making them fully differentiable within the TensorFlow graph.

### 2. Saliency Calculation
- We compute the gradient of the predicted class probability with respect to the 25 input PCA features: $
abla_{feat} P(y|X)$.
- **Back-projection:** Using the PCA components matrix $V$ (where $X_{pca} = X_{raw} V^T$), we map the gradients back to the 28x28 image space: $
abla_{raw} = 
abla_{pca} V$.
- **Visualization:** The resulting 28x28 heatmap shows which pixels in the original image "excited" the quantum circuit most during classification.

## Running the Saliency Demo (Docker)

To generate saliency maps for the top-2 categories:
```bash
docker run --rm -v $(pwd)/phasesix/results:/app/phasesix/results quantum-plankton python phasesix/quantum_saliency.py
```

## Expected Results
Check the `phasesix/results/` directory for:
- `saliency_example_0.png` through `saliency_example_4.png`: These files show the Original Image, the Quantum Saliency Heatmap, and an Overlay of the two.
