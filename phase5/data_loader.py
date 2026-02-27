import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def get_top_k_categories(k, data_dir=None):
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
    
    cats = []
    for c in os.listdir(data_dir):
        path = os.path.join(data_dir, c, 'training_data')
        if os.path.isdir(path):
            cats.append((c, len(os.listdir(path))))
    
    # Sort by count descending
    sorted_cats = sorted(cats, key=lambda x: x[1], reverse=True)
    return [c[0] for c in sorted_cats[:k]]

def load_plankton_k_categories(categories, img_size=(28, 28), data_dir=None):
    """Loads specific categories of plankton data."""
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
    
    X = []
    y = []
    
    for idx, cls in enumerate(categories):
        path = os.path.join(data_dir, cls, 'training_data')
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue
        
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(path, img_name)
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(img_size)
                    X.append(np.array(img))
                    y.append(idx)
                except Exception as e:
                    pass
                    
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def apply_pca_reduction(X_train, X_test, n_components=16):
    """
    Applies PCA to reduce dimensionality of images and scales to [0, 1].
    Expects X to be (N, H, W) or (N, D).
    """
    # Flatten images if they are 2D/3D
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)
    
    # Use PCA to extract the most informative features
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    # Scale to [0, 1] for Quantum Gate rotation (Ry(pi * value))
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    return X_train_scaled, X_test_scaled, pca
