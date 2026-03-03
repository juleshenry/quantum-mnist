import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold


def _sorted_image_files(directory):
    """Return sorted list of image filenames in *directory*.

    Sorting guarantees deterministic ordering across operating systems
    and filesystems, which is critical for reproducible train/test splits.
    """
    if not os.path.exists(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpeg', '.jpg', '.png'))
    )


def load_plankton_data(img_size=(128, 128), data_dir=None, test_size=0.15, val_size=0.15):
    """Loads all classes from the data directory (RGB, multi-class)."""
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    X, y = [], []
    for cls in classes:
        path = os.path.join(data_dir, cls, 'training_data')
        for img_name in _sorted_image_files(path):
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

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=42, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test, classes


def load_plankton_binary(class_a, class_b, img_size=(128, 128), data_dir=None,
                         random_state=42, max_per_class=None):
    """Load two plankton classes for binary classification.

    Parameters
    ----------
    class_a, class_b : str
        Names of the two plankton classes.
    img_size : tuple
        Target (height, width) for resizing.
    data_dir : str or None
        Root data directory.  Defaults to DATA_DIR env var.
    random_state : int
        Seed for the train/test split.
    max_per_class : int or None
        If set, cap the number of images loaded per class.  When used
        together with ``load_plankton_binary_kfold``, this ensures all
        models (CNN, Fair Classical, QNN) train on the same data budget.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
    """
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')

    def get_images(class_name):
        path = os.path.join(data_dir, class_name, 'training_data')
        images = []
        for img_name in _sorted_image_files(path):
            img_path = os.path.join(path, img_name)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                images.append(np.array(img))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
            if max_per_class is not None and len(images) >= max_per_class:
                break
        return np.array(images)

    imgs_a = get_images(class_a)
    imgs_b = get_images(class_b)

    if len(imgs_a) == 0 or len(imgs_b) == 0:
        raise ValueError(f"One of the classes {class_a} or {class_b} has no images.")

    print(f"  Loaded {class_a}: {len(imgs_a)} images, {class_b}: {len(imgs_b)} images")

    X = np.concatenate([imgs_a, imgs_b], axis=0).astype('float32') / 255.0
    y = np.concatenate([np.ones(len(imgs_a)), np.zeros(len(imgs_b))], axis=0)

    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


def load_plankton_binary_all(class_a, class_b, img_size=(128, 128), data_dir=None,
                             max_per_class=None):
    """Load all images for two classes without splitting.

    Returns the full X, y arrays.  The caller is responsible for
    splitting (e.g. via ``StratifiedKFold``).

    Returns
    -------
    X : ndarray of shape (N, H, W)
    y : ndarray of shape (N,) with values in {0, 1}
    """
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')

    def get_images(class_name):
        path = os.path.join(data_dir, class_name, 'training_data')
        images = []
        for img_name in _sorted_image_files(path):
            img_path = os.path.join(path, img_name)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                images.append(np.array(img))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
            if max_per_class is not None and len(images) >= max_per_class:
                break
        return np.array(images)

    imgs_a = get_images(class_a)
    imgs_b = get_images(class_b)

    if len(imgs_a) == 0 or len(imgs_b) == 0:
        raise ValueError(f"One of the classes {class_a} or {class_b} has no images.")

    print(f"  Loaded {class_a}: {len(imgs_a)} images, {class_b}: {len(imgs_b)} images")

    X = np.concatenate([imgs_a, imgs_b], axis=0).astype('float32') / 255.0
    y = np.concatenate([np.ones(len(imgs_a)), np.zeros(len(imgs_b))], axis=0)
    return X, y


def get_kfold_splitter(n_folds=5, random_state=42):
    """Return a StratifiedKFold splitter instance."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)


if __name__ == "__main__":
    # Quick verification
    X_train, X_test, y_train, y_test = load_plankton_binary(
        'dinobryon', 'nauplius', img_size=(4, 4))
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Shape: {X_train.shape}")
    print(f"Value range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"Train class balance: {np.mean(y_train):.3f}")
    print(f"Test  class balance: {np.mean(y_test):.3f}")

    # Verify determinism
    X2, _, _, _ = load_plankton_binary('dinobryon', 'nauplius', img_size=(4, 4))
    assert np.array_equal(X_train, X2), "Data loading is non-deterministic!"
    print("Determinism check: PASSED")
