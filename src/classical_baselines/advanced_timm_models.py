import os
import torch
import timm
import random
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

# -------------------------
# Config
# -------------------------
DATA_DIR = "ZooLake/images"   # root of class subfolders
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------------
# Data transforms
# -------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# Dataset + Loaders
# -------------------------
trainset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
valset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transforms)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_classes = len(trainset.classes)
print(f"Num classes: {num_classes}")

# -------------------------
# Model (transfer learning)
# -------------------------
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)

# -------------------------
# Train Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "zoolake_efficientnet_b0.pth")
