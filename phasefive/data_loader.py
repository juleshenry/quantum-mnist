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


def get_top_k_categories(k, data_dir=None):
    """Return names of the *k* most-populated plankton categories.

    Categories are counted by number of image files in their
    ``training_data/`` subdirectory, and ties are broken alphabetically
    for determinism.
    """
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')

    cats = []
    for c in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, c, 'training_data')
        if os.path.isdir(path):
            cats.append((c, len(_sorted_image_files(path))))

    # Primary sort: count descending.  Secondary sort: name ascending (stable sort).
    sorted_cats = sorted(cats, key=lambda x: (-x[1], x[0]))
    selected = [c[0] for c in sorted_cats[:k]]
    print(f"  Top-{k} categories: {selected}  (counts: {[c[1] for c in sorted_cats[:k]]})")
    return selected


def load_plankton_k_categories(categories, img_size=(28, 28), data_dir=None,
                               max_per_class=None):
    """Load specific categories of plankton data (grayscale) with a single split.

    Parameters
    ----------
    categories : list of str
    img_size : tuple
    data_dir : str or None
    max_per_class : int or None
        If set, cap the number of images loaded per class.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')

    X, y = _load_categories_raw(categories, img_size, data_dir, max_per_class)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def load_plankton_k_all(categories, img_size=(28, 28), data_dir=None,
                        max_per_class=None):
    """Load all images for *categories* without splitting.

    Returns the full X, y arrays.  The caller is responsible for
    splitting (e.g. via ``StratifiedKFold``).

    Returns
    -------
    X : ndarray of shape (N, H, W)
    y : ndarray of shape (N,) with integer class labels 0..k-1
    """
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
    return _load_categories_raw(categories, img_size, data_dir, max_per_class)


def _load_categories_raw(categories, img_size, data_dir, max_per_class):
    """Internal: load images for *categories* into arrays."""
    X, y = [], []

    for idx, cls in enumerate(categories):
        path = os.path.join(data_dir, cls, 'training_data')
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue

        count = 0
        for img_name in _sorted_image_files(path):
            img_path = os.path.join(path, img_name)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                X.append(np.array(img))
                y.append(idx)
                count += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
            if max_per_class is not None and count >= max_per_class:
                break
        print(f"  Class {idx} ({cls}): {count} images loaded")

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    return X, y


def get_kfold_splitter(n_folds=5, random_state=42):
    """Return a StratifiedKFold splitter instance."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
