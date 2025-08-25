import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
TRAIN_DIR = "data/archive/processed_dataset/train"
VAL_DIR = "data/archive/processed_dataset/val"
TEST_DIR = "data/archive/processed_dataset/test"
NEW_DATA_DIR = "data/new_waste"

BATCH_SIZE = 32
EPOCHS = 5  # adjust as needed

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

def get_data_loaders():
    """Load TrashNet + new_waste datasets with transforms"""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Base datasets (TrashNet)
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    # Add new data if available
    if os.path.exists(NEW_DATA_DIR) and any(os.scandir(NEW_DATA_DIR)):
        try:
            new_data = datasets.ImageFolder(NEW_DATA_DIR, transform=transform)
            print(f"📂 Found {len(new_data)} new samples, merging with TrashNet...")
            train_dataset = ConcatDataset([train_dataset, new_data])
        except Exception as e:
            print(f"⚠️ Skipping new_waste dataset due to error: {e}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def build_model(num_classes):
    """Load pretrained ResNet18 and modify classifier"""
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extractor

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(data_dir=TRAIN_DIR, save_path="models/latest_model.pth"):
    """Train model using TrashNet + new submissions"""

    print("📊 Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders()

    num_classes = len(os.listdir(TRAIN_DIR))
    model = build_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"✅ Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
              f"Accuracy={(100*correct/total):.2f}%")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")

    return model
