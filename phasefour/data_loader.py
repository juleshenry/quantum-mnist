import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_plankton_binary(class_a, class_b, img_size=(28, 28), data_dir='data/zooplankton_0p5x'):
    def get_images(class_name):
        path = os.path.join(data_dir, class_name, 'training_data')
        images = []
        if not os.path.exists(path):
            return np.array([])
        for img_name in os.listdir(path):
            if img_name.endswith('.jpeg'):
                img_path = os.path.join(path, img_name)
                try:
                    img = Image.open(img_path).convert('L') # Grayscale
                    img = img.resize(img_size)
                    images.append(np.array(img) / 255.0)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return np.array(images)

    imgs_a = get_images(class_a)
    imgs_b = get_images(class_b)

    X = np.concatenate([imgs_a, imgs_b], axis=0)
    y = np.concatenate([np.ones(len(imgs_a)), np.zeros(len(imgs_b))], axis=0)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Test loading
    X_train, X_test, y_train, y_test = load_plankton_binary('dinobryon', 'nauplius', img_size=(4, 4))
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Shape: {X_train.shape}")
