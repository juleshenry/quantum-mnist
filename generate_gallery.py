import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from phase5.data_loader import get_top_k_categories, _load_categories_raw

def generate_preprocessing_gallery():
    os.makedirs('results/preprocessing', exist_ok=True)
    
    k = 4
    categories = get_top_k_categories(k)
    print(f"Categories for gallery: {categories}")
    
    # Load raw images (28x28 grayscale)
    X, y = _load_categories_raw(categories, img_size=(28, 28), data_dir='data/zooplankton_0p5x', max_per_class=10)
    
    # We want to show: Original -> Downsampled (4x4)
    # The project uses PCA mostly, but for "visual" preprocessing we often show downsampling.
    # Let's show a few examples
    n_examples = 4
    fig, axes = plt.subplots(n_examples, 2, figsize=(6, 2 * n_examples))
    
    plt.suptitle("Image Preprocessing: Original (28x28) vs. Downsampled (4x4)", fontsize=14)
    
    for i in range(n_examples):
        # Pick one image from each category
        idx = np.where(y == i)[0][0]
        img_28 = X[idx]
        
        # Manually downsample to 4x4 for visualization
        img_pil = Image.fromarray((img_28 * 255).astype(np.uint8))
        img_4 = np.array(img_pil.resize((4, 4), resample=Image.BOX)) / 255.0
        
        # Plot Original
        axes[i, 0].imshow(img_28, cmap='gray')
        axes[i, 0].set_title(f"{categories[i]} (28x28)")
        axes[i, 0].axis('off')
        
        # Plot 4x4
        axes[i, 1].imshow(img_4, cmap='gray', interpolation='nearest')
        axes[i, 1].set_title("Downsampled (4x4)")
        axes[i, 1].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = 'results/preprocessing/preprocessing_gallery.png'
    plt.savefig(save_path)
    print(f"Gallery saved to {save_path}")

if __name__ == "__main__":
    generate_preprocessing_gallery()
