"""
Train script for TrashNet dataset (local version).
Fine-tunes ResNet18 and saves trained artifacts.

Run:
python scripts/train_model.py --data_dir "data/archive/dataset-resized" --epochs 8 --batch_size 32
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())


# ------------------------ CONFIG ------------------------
ARTIFACT_DIR = "models/artifacts"
CLASS_MAP = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# --------------------------------------------------------


def split_train_val_test(src_root, target_root, train=0.7, val=0.15, test=0.15, seed=42):
    """
    Split dataset into train/val/test from original dataset-resized folder.
    """
    random.seed(seed)
    src_root = Path(src_root)
    target_root = Path(target_root)
    for split in ["train", "val", "test"]:
        for cls in CLASS_MAP:
            (target_root / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASS_MAP:
        items = list((src_root / cls).glob("*.*"))
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train)
        n_val = int(n * val)
        splits = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:]
        }
        for split, files in splits.items():
            for f in files:
                dst = target_root / split / cls / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)


def build_loaders(base_dir, batch_size=32):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(base_dir, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(base_dir, "test"), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, train_ds.classes


def train(model, train_loader, val_loader, device, epochs=8, lr=3e-4, wd=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler()
    best_val = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running / total
        train_acc = correct / total

        # Validation
        model.eval()
        vtotal, vcorrect, vrunning = 0, 0, 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} - val"):
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                vrunning += loss.item() * x.size(0)
                preds = logits.argmax(1)
                vcorrect += (preds == y).sum().item()
                vtotal += y.size(0)

        val_loss = vrunning / vtotal if vtotal else 0.0
        val_acc = vcorrect / vtotal if vtotal else 0.0
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

    return best_state, best_val


def evaluate(model, test_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total else 0.0
    print(f"✅ Test accuracy: {acc:.3f}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset-resized folder")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    base_dir = data_dir.parent / "processed_dataset"

    # Step 1: Split into train/val/test if not already done
    if not (base_dir / "train").exists():
        print("📂 Preparing dataset splits...")
        split_train_val_test(data_dir, base_dir)
    else:
        print("✅ Found existing processed dataset.")

    # Step 2: Dataloaders
    train_loader, val_loader, test_loader, classes = build_loaders(base_dir, batch_size=args.batch_size)
    print(f"Classes: {classes}")

    # Step 3: Model
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("⚠️ Training on CPU (much slower)")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    # Step 4: Train
    best_state, best_val = train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
    print(f"Best val acc: {best_val:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Step 5: Evaluate
    test_acc = evaluate(model, test_loader, device)

    # Step 6: Save artifacts
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "best_model.pt"))
    label_map = {str(i): cls for i, cls in enumerate(classes)}
    with open(os.path.join(ARTIFACT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "training_summary.txt"), "w") as f:
        f.write(f"Best Val Acc: {best_val:.4f}\nTest Acc: {test_acc:.4f}\n")


if __name__ == "__main__":
    main()
