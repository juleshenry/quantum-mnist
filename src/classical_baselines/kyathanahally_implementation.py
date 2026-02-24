"""
Deep Learning Classification of Lake Zooplankton
Implementation based on: https://arxiv.org/abs/2108.05258
Kyathanahally et al., 2021

Optimized for Apple M1 (MPS backend) with PyTorch.
Uses transfer learning with EfficientNet, DenseNet121, MobileNet, ResNet50, InceptionV3.
Supports average and stacking ensemble methods.

─── Quick start ─────────────────────────────────────────────────────
# Install deps:
pip install torch torchvision timm scikit-learn pandas matplotlib pillow tqdm

# Demo (no data needed — synthetic images):
python zooplankton_classifier.py --demo

# Real ZooLake data (download from https://data.eawag.ch/dataset/...):
python zooplankton_classifier.py --data_dir /path/to/ZooLake

# Choose specific architectures:
python zooplankton_classifier.py --data_dir /path/to/ZooLake \
    --archs mobilenet densenet121 efficientnet_b2 efficientnet_b7
─────────────────────────────────────────────────────────────────────
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("⚠️  timm not installed. EfficientNet unavailable. Install: pip install timm")

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Device — auto-detect M1 MPS / CUDA / CPU
# ─────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("✅ Apple M1/M2 MPS (Metal Performance Shaders) detected")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU detected")
        return torch.device("cuda")
    else:
        print("⚠️  No GPU found — using CPU")
        return torch.device("cpu")


DEVICE = get_device()

# ─────────────────────────────────────────────────────────────
# Hyperparameters (paper defaults)
# ─────────────────────────────────────────────────────────────
SEED         = 42
IMG_SIZE     = 128      # Paper: 128×128 pixels
BATCH_SIZE   = 32
EPOCHS_HEAD  = 50       # Phase-1 max epochs (backbone frozen)
EPOCHS_FT    = 20       # Phase-2 fine-tuning epochs (reduced for demo; paper uses 400)
LR_HEAD      = 1e-3     # Head training LR
LR_FINETUNE  = 1e-7     # Fine-tune LR (paper's value)
EARLY_STOP   = 10       # Patience (paper uses 50; shortened for practical demo)
DROPOUT      = 0.3
N_CLASSES    = 35       # ZooLake: 35 plankton classes

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────
# Transforms — matching paper's augmentation strategy
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),            # Method (i) from paper: resize ignoring aspect ratio
    transforms.RandomRotation(180),                     # Paper: rotations up to 180°
    transforms.RandomHorizontalFlip(),                  # Paper: flipping
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, shear=10,        # Paper: shear up to 10%
                            scale=(0.8, 1.2)),          # Paper: zoom up to 20%
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class ZooLakeDataset(Dataset):
    """
    Folder-based dataset for ZooLake or any image-per-class structure:

        root/
            dinobryon/
                img001.jpg ...
            daphnia/
                img001.jpg ...
            ...

    ZooLake download:
        https://data.eawag.ch/dataset/deep-learning-classification-of-zooplankton-from-lakes
    """

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []
        for cls in self.classes:
            for p in (self.root / cls).glob("*"):
                if p.suffix.lower() in self.VALID_EXT:
                    self.samples.append((str(p), self.class_to_idx[cls]))
        print(f"  → {len(self.samples)} images | {len(self.classes)} classes | root={root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_dataloaders(data_dir: str, batch_size: int = BATCH_SIZE):
    """Split dataset 70 / 15 / 15 (train / val / test) as in the paper."""
    full = ZooLakeDataset(data_dir)
    n = len(full)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    indices = list(range(n))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    tr_idx = indices[:n_train]
    va_idx = indices[n_train:n_train + n_val]
    te_idx = indices[n_train + n_val:]

    # Wrap with transforms
    tr_ds = ZooLakeDataset(data_dir, transform=train_transform)
    va_ds = ZooLakeDataset(data_dir, transform=val_transform)
    te_ds = ZooLakeDataset(data_dir, transform=val_transform)

    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(tr_ds, tr_idx), batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(Subset(va_ds, va_idx), batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(Subset(te_ds, te_idx), batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader, full.classes


# ─────────────────────────────────────────────────────────────
# Model Factory — Transfer Learning
# ─────────────────────────────────────────────────────────────
def build_model(arch: str, num_classes: int,
                dropout: float = DROPOUT,
                freeze_backbone: bool = True) -> nn.Module:
    """
    Return a pretrained ImageNet model with a custom dropout+linear head.
    Paper's approach: freeze all → add head → Phase-1 → unfreeze all → Phase-2.
    """
    arch_l = arch.lower()

    def _freeze_except(model, *keep_names):
        for name, p in model.named_parameters():
            if not any(k in name for k in keep_names):
                p.requires_grad = False

    # ── EfficientNet B0–B7 (timm) ─────────────────────────
    if arch_l.startswith("efficientnet"):
        if not HAS_TIMM:
            raise ImportError("timm is required for EfficientNet. pip install timm")
        model = timm.create_model(arch_l, pretrained=True, num_classes=0, drop_rate=dropout)
        feat_dim = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        if freeze_backbone:
            _freeze_except(model, "classifier")

    # ── DenseNet121 ────────────────────────────────────────
    elif arch_l == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        feat_dim = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        if freeze_backbone:
            _freeze_except(model, "classifier")

    # ── MobileNetV2 ────────────────────────────────────────
    elif arch_l == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        if freeze_backbone:
            _freeze_except(model, "classifier")

    # ── ResNet50 ───────────────────────────────────────────
    elif arch_l == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        if freeze_backbone:
            _freeze_except(model, "fc")

    # ── InceptionV3 ────────────────────────────────────────
    elif arch_l == "inceptionv3":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        feat_dim = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        if freeze_backbone:
            _freeze_except(model, "fc", "AuxLogits")

    else:
        raise ValueError(
            f"Unknown arch '{arch}'. Choose from:\n"
            "  efficientnet_b0..b7 (requires timm), densenet121,\n"
            "  mobilenet, resnet50, inceptionv3"
        )

    return model


# ─────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int, save_path: str):
        self.patience  = patience
        self.save_path = save_path
        self.best      = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def _run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False, desc="  train" if training else "  eval "):
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            out = model(imgs)
            # InceptionV3 returns (logits, aux) during training
            if isinstance(out, tuple):
                loss = criterion(out[0], labels) + 0.4 * criterion(out[1], labels)
                logits = out[0]
            else:
                loss = criterion(out, labels)
                logits = out
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
            if not training:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total
    acc = correct / total
    if training:
        return avg_loss, acc
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ─────────────────────────────────────────────────────────────
# Two-phase training (paper §II.G)
# ─────────────────────────────────────────────────────────────
def train_model(
    arch: str,
    train_loader, val_loader,
    num_classes: int = N_CLASSES,
    save_dir: str = "checkpoints",
    device: torch.device = DEVICE,
):
    """
    Phase 1: Frozen backbone → train head (LR=1e-3, early-stop)
    Phase 2: Unfreeze all   → fine-tune   (LR=1e-7, early-stop)
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f"{arch}_best.pt")
    model = build_model(arch, num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    # ── Phase 1 ──────────────────────────────────────────
    print(f"\n[Phase 1] Backbone frozen — training head for {arch}")
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    es = EarlyStopping(patience=EARLY_STOP, save_path=ckpt)

    for ep in range(1, EPOCHS_HEAD + 1):
        tr_loss, _    = _run_epoch(model, train_loader, criterion, opt, device, training=True)
        vl_loss, acc, f1 = _run_epoch(model, val_loader, criterion, None, device, training=False)
        sched.step(vl_loss)
        es(vl_loss, model)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        print(f"  Ep {ep:3d}/{EPOCHS_HEAD} | tr={tr_loss:.4f} vl={vl_loss:.4f} "
              f"acc={acc:.4f} F1={f1:.4f}" + (" ← best" if es.counter == 0 else ""))
        if es.stop:
            print(f"  Early stop at ep {ep}")
            break

    # ── Phase 2 ──────────────────────────────────────────
    print(f"\n[Phase 2] Unfreeze all — fine-tuning {arch} at LR={LR_FINETUNE}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    for p in model.parameters():
        p.requires_grad = True
    opt_ft = optim.Adam(model.parameters(), lr=LR_FINETUNE)
    es_ft  = EarlyStopping(patience=EARLY_STOP, save_path=ckpt)

    for ep in range(1, EPOCHS_FT + 1):
        tr_loss, _    = _run_epoch(model, train_loader, criterion, opt_ft, device, training=True)
        vl_loss, acc, f1 = _run_epoch(model, val_loader, criterion, None, device, training=False)
        es_ft(vl_loss, model)
        print(f"  FT {ep:3d}/{EPOCHS_FT} | vl={vl_loss:.4f} acc={acc:.4f} F1={f1:.4f}"
              + (" ← best" if es_ft.counter == 0 else ""))
        if es_ft.stop:
            print(f"  Early stop at FT ep {ep}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, history


# ─────────────────────────────────────────────────────────────
# Ensemble (paper §II.H)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def confidence_vectors(model: nn.Module, loader, device) -> tuple:
    """Return softmax probs (N, C) and true labels (N,)."""
    model.eval()
    sm = nn.Softmax(dim=1)
    probs_list, label_list = [], []
    for imgs, labels in tqdm(loader, leave=False, desc="  probs"):
        out = model(imgs.to(device))
        logits = out[0] if isinstance(out, tuple) else out
        probs_list.append(sm(logits).cpu().numpy())
        label_list.extend(labels.numpy())
    return np.vstack(probs_list), np.array(label_list)


def average_ensemble(model_list: list, loader, device):
    """Average confidence vectors (Section II.H.1)."""
    all_probs, labels = [], None
    for m in model_list:
        p, y = confidence_vectors(m, loader, device)
        all_probs.append(p)
        labels = y
    return np.mean(all_probs, axis=0).argmax(axis=1), labels


def stacking_ensemble(model_list: list, train_loader, eval_loader, device):
    """Logistic regression meta-learner on confidence vectors (Section II.H.2)."""
    print("  Building stacking meta-dataset...")
    X_tr, y_tr = [], None
    for m in model_list:
        p, y = confidence_vectors(m, train_loader, device)
        X_tr.append(p); y_tr = y
    X_ev, y_ev = [], None
    for m in model_list:
        p, y = confidence_vectors(m, eval_loader, device)
        X_ev.append(p); y_ev = y
    print("  Fitting logistic regression meta-learner...")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                             multi_class="multinomial", n_jobs=-1)
    clf.fit(np.hstack(X_tr), y_tr)
    preds = clf.predict(np.hstack(X_ev))
    return preds, y_ev


# ─────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────
def evaluate(preds, labels, class_names, title=""):
    acc = (preds == labels).mean()
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"{'─'*55}")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    return acc, f1


def plot_history(history: dict, arch: str, save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_acc"]);   axes[1].set_title("Val Accuracy")
    axes[2].plot(history["val_f1"]);    axes[2].set_title("Val Macro F1")
    plt.suptitle(f"Training — {arch}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{arch}_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


# ─────────────────────────────────────────────────────────────
# MLP on Morphological Features (paper's "Feature" model)
# ─────────────────────────────────────────────────────────────
class MorphMLP(nn.Module):
    """
    MLP on 110 morphological + color features (paper Table I: Feature model, ~91% acc).
    To use: compute features externally, normalize to zero-mean unit-variance,
    then call this model with a (batch, 110) tensor.
    """
    def __init__(self, input_dim: int = 110,
                 num_classes: int = N_CLASSES,
                 dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Synthetic Demo (no dataset required)
# ─────────────────────────────────────────────────────────────
def run_demo():
    """Smoke-test the full pipeline with random images (5 classes)."""
    import tempfile, shutil
    print("\n" + "="*60)
    print("  DEMO: synthetic data, 5 classes × 20 images, MobileNet only")
    print("="*60)
    tmp = tempfile.mkdtemp()
    try:
        for cls_i in range(5):
            (Path(tmp) / f"class_{cls_i}").mkdir()
            for j in range(20):
                arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(arr).save(Path(tmp) / f"class_{cls_i}" / f"img_{j:03d}.jpg")

        train_l, val_l, test_l, class_names = make_dataloaders(tmp, batch_size=8)
        model, history = train_model(
            "mobilenet", train_l, val_l,
            num_classes=len(class_names),
            save_dir="/tmp/zoo_ckpt",
            device=DEVICE,
        )
        plot_history(history, "mobilenet_demo", save_dir="/tmp/zoo_plots")
        probs, labels = confidence_vectors(model, test_l, DEVICE)
        evaluate(probs.argmax(1), labels, class_names, title="MobileNet (demo)")
        print("\n✅ Demo complete — pipeline works on this machine.")
    finally:
        shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────
# Full pipeline on real ZooLake data
# ─────────────────────────────────────────────────────────────
def run_full(data_dir: str, archs: Optional[list] = None):
    """
    Train multiple architectures, then combine with average + stacking ensemble.
    Default 'Best 6' from the paper (requires timm for EfficientNet).
    """
    if archs is None:
        archs = ["densenet121", "mobilenet", "resnet50"]
        if HAS_TIMM:
            archs += ["efficientnet_b2", "efficientnet_b5", "efficientnet_b7"]

    train_l, val_l, test_l, class_names = make_dataloaders(data_dir)
    nc = len(class_names)
    print(f"\nClasses ({nc}): {class_names}\n")

    trained, results = [], {}
    for arch in archs:
        print(f"\n{'='*60}\n  Training: {arch}\n{'='*60}")
        m, h = train_model(arch, train_l, val_l, num_classes=nc, device=DEVICE)
        trained.append(m)
        plot_history(h, arch)
        p, y = confidence_vectors(m, test_l, DEVICE)
        acc, f1 = evaluate(p.argmax(1), y, class_names, title=f"{arch} (single)")
        results[arch] = {"accuracy": acc, "f1": f1}

    if len(trained) > 1:
        print(f"\n{'='*60}\n  Average Ensemble\n{'='*60}")
        ep, ey = average_ensemble(trained, test_l, DEVICE)
        acc, f1 = evaluate(ep, ey, class_names, title="Average Ensemble")
        results["avg_ensemble"] = {"accuracy": acc, "f1": f1}

        print(f"\n{'='*60}\n  Stacking Ensemble\n{'='*60}")
        sp, sy = stacking_ensemble(trained, train_l, test_l, DEVICE)
        acc, f1 = evaluate(sp, sy, class_names, title="Stacking Ensemble")
        results["stack_ensemble"] = {"accuracy": acc, "f1": f1}

    df = pd.DataFrame(results).T
    df.index.name = "model"
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(df.to_string())
    df.to_csv("results_summary.csv")
    print("\nSaved → results_summary.csv")
    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Lake Zooplankton Classifier — Kyathanahally et al. 2021",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zooplankton_classifier.py --demo
  python zooplankton_classifier.py --data_dir ./ZooLake
  python zooplankton_classifier.py --data_dir ./ZooLake --archs mobilenet densenet121
        """)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset folder (class-per-subfolder structure)")
    parser.add_argument("--archs", nargs="+", default=None,
                        help="Architectures to train (default: paper's Best 6)")
    parser.add_argument("--demo", action="store_true",
                        help="Run synthetic demo (no dataset needed)")
    args = parser.parse_args()

    if args.demo or args.data_dir is None:
        run_demo()
    else:
        run_full(args.data_dir, args.archs)

