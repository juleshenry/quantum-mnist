import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def load_plankton_data(img_size=(128, 128), data_dir=None, test_size=0.15, val_size=0.15):
    """Loads all classes from the data directory."""
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    X = []
    y = []
    
    for cls in classes:
        path = os.path.join(data_dir, cls, 'training_data')
        if not os.path.exists(path):
            continue
        
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    X.append(np.array(img))
                    y.append(class_to_idx[cls])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    # Split into train, val, test (70:15:15 as per paper)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Calculate relative val_size for the remaining train_val set
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_relative_size, random_state=42, stratify=y_train_val)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, classes

def load_plankton_binary(class_a, class_b, img_size=(128, 128), data_dir=None, random_state=42):
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
    def get_images(class_name):
        path = os.path.join(data_dir, class_name, 'training_data')
        images = []
        if not os.path.exists(path):
            return np.array([])
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(path, img_name)
                try:
                    # Convert to grayscale for both classical and quantum consistency
                    img = Image.open(img_path).convert('L')
                    img = img.resize(img_size)
                    images.append(np.array(img))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return np.array(images)

    imgs_a = get_images(class_a)
    imgs_b = get_images(class_b)

    if len(imgs_a) == 0 or len(imgs_b) == 0:
        raise ValueError(f"One of the classes {class_a} or {class_b} has no images.")

    X = np.concatenate([imgs_a, imgs_b], axis=0).astype('float32') / 255.0
    y = np.concatenate([np.ones(len(imgs_a)), np.zeros(len(imgs_b))], axis=0)

    # Use stratified split for better rigor
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

def apply_pca_reduction(X_train, X_test, n_components=16):
    """Applies PCA to reduce dimensionality and scales to [0, 1]."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    # Flatten if necessary
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)
    
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    return X_train_scaled, X_test_scaled, pca

if __name__ == "__main__":
    # Test loading
    X_train, X_test, y_train, y_test = load_plankton_binary('dinobryon', 'nauplius', img_size=(28, 28))
    X_train_p, X_test_p, _ = apply_pca_reduction(X_train, X_test, n_components=16)
    print(f"Train size: {len(X_train_p)}, Features: {X_train_p.shape[1]}")
