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
import time
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_loader import load_plankton_data
from classical_models import SmallCNN, create_transfer_mobilenet

def train_model(model, train_loader, val_loader, device, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    return history

def prepare_data_loaders(X, y, batch_size=32):
    # PyTorch expects (N, C, H, W)
    X = np.transpose(X, (0, 3, 1, 2))
    tensor_x = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_and_evaluate():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_plankton_data(img_size=(128, 128), data_dir='../data/zooplankton_0p5x')
    
    num_classes = len(classes)
    print(f"Loaded {len(X_train)} training images, {len(X_val)} validation, {len(X_test)} test.")
    
    train_loader = prepare_data_loaders(X_train, y_train)
    val_loader = prepare_data_loaders(X_val, y_val)
    test_loader = prepare_data_loaders(X_test, y_test)

    results = []

    # 1. Small CNN
    print("\nTraining Small CNN...")
    small_cnn = SmallCNN(num_classes=num_classes).to(device)
    start_time = time.time()
    history_small = train_model(small_cnn, train_loader, val_loader, device, epochs=20)
    duration = time.time() - start_time
    
    # Evaluate
    small_cnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = small_cnn(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = correct / total
    
    results.append({
        'model': 'SmallCNN',
        'test_accuracy': float(test_acc),
        'training_time': float(duration),
        'num_params': sum(p.numel() for p in small_cnn.parameters())
    })
    print(f"SmallCNN Test Accuracy: {test_acc:.4f}")

    # 2. MobileNetV2 Transfer
    print("\nTraining MobileNetV2 (Transfer Learning)...")
    mobile_model = create_transfer_mobilenet(num_classes=num_classes).to(device)
    start_time = time.time()
    history_mobile = train_model(mobile_model, train_loader, val_loader, device, epochs=15)
    duration = time.time() - start_time
    
    # Evaluate
    mobile_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mobile_model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc_m = correct / total
    
    results.append({
        'model': 'MobileNetV2_Transfer',
        'test_accuracy': float(test_acc_m),
        'training_time': float(duration),
        'num_params': sum(p.numel() for p in mobile_model.parameters())
    })
    print(f"MobileNetV2 Test Accuracy: {test_acc_m:.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/classical_phase_four_results.csv', index=False)
    
    with open('results/training_histories.json', 'w') as f:
        json.dump({
            'small_cnn': history_small,
            'mobilenet': history_mobile
        }, f)

    print("\nResults saved to results/classical_phase_four_results.csv")

if __name__ == "__main__":
    train_and_evaluate()
