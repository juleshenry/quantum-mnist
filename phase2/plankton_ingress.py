"""
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
from sklearn.model_selection import train_test_split

# Configuration
QUBIT_DIMS = (4, 4)
DATA_DIR = Path(__file__).parent.parent / 'data' / 'zooplankton_0p5x'

def get_plankton_names():
    plank_dir = os.listdir(DATA_DIR)
    plank = sorted(filter(lambda x: not x.startswith('.'), plank_dir))
    return plank

def load_images_for_class(class_name, limit=50):
    class_dir = DATA_DIR / class_name / 'training_data'
    if not class_dir.exists():
        return []
    
    img_paths = list(class_dir.glob('*.jpeg'))[:limit]
    imgs = []
    for p in img_paths:
        try:
            with Image.open(p) as img:
                # Convert to grayscale and resize to 4x4 as per requirement
                img = img.convert('L')
                img = img.resize(QUBIT_DIMS, Image.BILINEAR)
                imgs.append(np.asarray(img) / 255.0)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return imgs

def prepare_binary_dataset(class_a, class_b, limit=50, test_split=0.25, seed=42):
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

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
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
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    plank = get_plankton_names()
    print(f"Found {len(plank)} classes.")
    if len(plank) >= 2:
        (x_train, y_train), (x_test, y_test) = prepare_binary_dataset(plank[0], plank[1])
        print(f"Prepared dataset for {plank[0]} vs {plank[1]}")
        print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
