import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
MODEL_PATH = "models/latest_model.pth"
TEST_DIR = r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_management_system\data\archive\dataset-resized"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
num_classes = len(test_dataset.classes)  # should be 6 with your dataset
model = models.resnet18(weights=None)   # don't load ImageNet weights here
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load checkpoint but drop incompatible classifier weights
state_dict = torch.load(MODEL_PATH, map_location=device)
state_dict.pop("fc.weight", None)
state_dict.pop("fc.bias", None)
model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.eval()

# Evaluate
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total if total > 0 else 0
print(f"Test Accuracy: {accuracy:.2f}%")
