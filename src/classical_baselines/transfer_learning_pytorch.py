import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Check for Apple Silicon GPU (MPS)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 1. Data Augmentation (Per Section 2.4 of the paper)
# They used 128x128 resizing, rotation up to 180, flipping, zooming, and shearing.
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Model Selection (Paper uses ResNet50 as a primary backbone)
def get_model(num_classes):
    # weights=models.ResNet50_Weights.DEFAULT is the modern API for pretrained=True
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze early layers if doing strict transfer learning
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

# 3. Training Loop (Optimized for MPS)
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f}')

# 4. Main Execution
if __name__ == "__main__":
    # Define your dataset path
    # DATA_DIR = 'path_to_zoolake_dataset'
    # full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    # train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    # Example initialization
    num_classes = 35 # As specified in the paper
    model = get_model(num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Model ready for training on M1 GPU.")
