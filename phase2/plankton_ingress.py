r"""
                               ★■╬▂▂▂▂▂◓□                                                           
                              ☆◕◓◊◊▇▅◕⬤▽■●⬤                                                         
                                   ▽■◑▅▆◑★■╬◒.                                                      
                                       ⬤▂▄◔▽▽▅◒◕★                                                   
                                         ⬤▄▄○◈◔◓◊○⬤▼                                                
                                          .▲◈█▆▅▇██▇▂■                                              
                                             ☆☆■◑█▇███○                                             
                                         ★★★★.  □█▲○██▄                                             
                                        ◒▇╬◑●◊■□▂▲★███◐                                             
                                        ■○╬╬■▇▄□▂ ☆◊██△                                             
                                          ★▆█▅█○▂ ▼◈█▇◑                                             
                                           ▲█★██▂  ☆██▆                                             
                                           ▼█◐◐█▂  ▲██▆                                             
                                           ★█□☆▼▽  ▲██▄                                             
                                           ★█◓     ▲██▄                                             
                                           ▲█◒     ▽◈█▆◕★                                           
                                          ◓▄○.      ▽▅██▲                                           
                                         ☆◒▅   ▽□□□■.□▇█▇△.                                         
                                         ◊○⬤   ⬤▅▄▄◊  □███▂△                                        
                                         ◊◊▄           ◑███▄△                                       
                                        ◓▇◒△           △▇███◒                                       
                                        ◐█◒            ▽△◒██▂                                       
                                        ◕█◒    ☆○◈◈◈     ◒██▂                                       
                                        ◕█◒    ▲████★    ◒██▂                                       
                                       △●○▼    ▲████★  . ◒██▅△                                      
                                     ★□▂▅◕     ▲████★  . ◒███▆△▽                                    
                                    ◓◈◊◕◑     ☆▲████▅◈   ◓█████▂◑★                                  
                                   ◕◊▼  ◒      ▲█████▂   ◓▆██████◐☆                                 
                                  ●▆▽△▄▇●      ▲████▇╬   □◊███████▂                                 
                                  ▽█.◑██●     .▼███◈◑◒   △⬤███████⬤                                 
                                   ▆☆◑██●      .▽▽▽       ■███▄███▼                                 
                                   ◔●◐▲▄●          .      ■██▇▼╬▄◐★                                 
                                      ▆█●                 ▲╬██▂☆.                                   
                                      ▆█●              .  ▼▇██▄                                     
                                  .☆□●▇●◕                 ▽▆██▇◊■☆.                                 
                                 ▽◑○⬤▼╬○△                 .□████▆█◑▲.                               
                                ⬤◈★▽◔▅█╬▼                   ████▇███◐☆                              
                              .⬤●▽△○█▇██○▲                 ■████▂████◒▽                             
                              □▆.▼▂▅⬤╬██◕    ☆☆★★★★★☆★★.  .▂████◒◓████●                             
                             □◐▽☆◈○▲▂██◑●◓▼▲◓██████████╬◒◊███████◈□███▇◔                            
                             ⬤●□█◊▼☆◐▆█◐◓██████████████████████▆●△☆◔███◒                            
                             ⬤▆▄█⬤   ☆▂████◕◓◈◈◈◈◈◓■◈◈◈◈◒○███▆⬤▽    ▅██◒                            
                             ▽◒◊◒★    .◔▅█▇▲             ◓█▇●▲      ⬤╬◐▼                            
                                        ☆▲★               ▼▼                                        
                                                    /               
                                ___       ___  ___ (___       _ _   
                                |   )|   )|   )|   )|    |   )| | )  
                                |__/||__/ |__/||  / |__  |__/ |  /   
                                    |                                
                                                                    
                                    /           /    /             
                                ___ (  ___  ___ (    (___  ___  ___ 
                                |   )| |   )|   )|___)|    |   )|   )
                                |__/ | |__/||  / | \  |__  |__/ |  / 
                                |                                    
                                                                    
                                                /    /               
                                _ _  ___  ___ (___    ___  ___      
                                | | )|   )|    |   )| |   )|___)     
                                |  / |__/||__  |  / | |  / |__       
                                                                    
                                                                    
                                /                     /             
                                (  ___  ___  ___  ___    ___  ___    
                                | |___)|   )|   )|   )| |   )|   )   
                                | |__  |__/||    |  / | |  / |__/    
                                                            __/                                                                                                         
                                                                                                    
                                            by Julian Henry                                                        
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configuration
LOAD_DIMS = (28, 28)      # Load at 28x28 to preserve spatial info for PCA
QUBIT_DIMS = (4, 4)       # Legacy — kept for reference / test compat
N_PCA_COMPONENTS = 16     # Must match number of data qubits
DATA_DIR = Path(__file__).parent.parent / 'data' / 'zooplankton_0p5x'

def get_plankton_names():
    plank_dir = os.listdir(DATA_DIR)
    plank = sorted(filter(lambda x: not x.startswith('.'), plank_dir))
    return plank

def load_images_for_class(class_name, limit=50, dims=None):
    """Load images for a class, resized to *dims* (default LOAD_DIMS=28x28).

    Returns a list of 2-D numpy arrays with pixel values in [0, 1].
    """
    if dims is None:
        dims = LOAD_DIMS
    class_dir = DATA_DIR / class_name / 'training_data'
    if not class_dir.exists():
        return []
    
    img_paths = sorted(class_dir.glob('*.jpeg'))[:limit]
    imgs = []
    for p in img_paths:
        try:
            with Image.open(p) as img:
                img = img.convert('L')
                img = img.resize(dims, Image.BILINEAR)
                imgs.append(np.asarray(img) / 255.0)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return imgs


def pca_transform(X_train, X_test, n_components=N_PCA_COMPONENTS, seed=42):
    """Apply PCA + MinMaxScaler (fit on train only, transform both).

    Parameters
    ----------
    X_train, X_test : ndarray of shape (N, H, W)
        Grayscale images in [0, 1].
    n_components : int
        Number of PCA components (should equal number of data qubits).
    seed : int
        Random seed for PCA reproducibility.

    Returns
    -------
    X_train_scaled, X_test_scaled : ndarray of shape (N, n_components)
        PCA-reduced features scaled to [0, 1], ready for Ry angle encoding.
    """
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    return X_train_scaled, X_test_scaled

def prepare_binary_dataset(class_a, class_b, limit=50, test_split=0.25, seed=42,
                           apply_pca=True):
    """Load two classes and split into train/test with stratification.

    Parameters
    ----------
    class_a, class_b : str
        Plankton class directory names.
    limit : int
        Maximum images per class.
    test_split : float
        Fraction held out for testing.
    seed : int or None
        Random seed for reproducible, stratified splitting.  Pass ``None``
        for legacy (non-deterministic) behaviour.
    apply_pca : bool
        If True (default), apply PCA + MinMaxScaler after splitting.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        When *apply_pca* is True, X arrays have shape (N, 16) with values
        in [0, 1] ready for Ry angle encoding.
    """
    imgs_a = load_images_for_class(class_a, limit)
    imgs_b = load_images_for_class(class_b, limit)

    labels_a = np.zeros(len(imgs_a))
    labels_b = np.ones(len(imgs_b))

    X = np.array(imgs_a + imgs_b)
    y = np.concatenate([labels_a, labels_b])

    # Stratified split keeps class balance in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )

    if apply_pca:
        X_train, X_test = pca_transform(X_train, X_test, seed=seed)

    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    plank = get_plankton_names()
    print(f"Found {len(plank)} classes.")
    if len(plank) >= 2:
        (x_train, y_train), (x_test, y_test) = prepare_binary_dataset(plank[0], plank[1])
        print(f"Prepared dataset for {plank[0]} vs {plank[1]}")
        print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
