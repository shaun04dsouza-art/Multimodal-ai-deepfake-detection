import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_DIR = r"D:\deepfake_detection_project\data\images"
SAVE_PATH = r"D:\deepfake_detection_project\models\image_detector\resnet18.pth"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Label mapping:", dataset.class_to_idx)
# Should be: {'fake': 0, 'real': 1}

# --------------------------------------------------
# CORRECT TRAIN/VAL SPLIT (NO LEAKAGE)
# --------------------------------------------------
fake_idx = [i for i, (_, label) in enumerate(dataset) if label == 0]
real_idx = [i for i, (_, label) in enumerate(dataset) if label == 1]

fake_train, fake_val = train_test_split(fake_idx, test_size=0.2, random_state=42)
real_train, real_val = train_test_split(real_idx, test_size=0.2, random_state=42)

train_idx = fake_train + real_train
val_idx = fake_val + real_val

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# --------------------------------------------------
# MODEL — RESNET18
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
EPOCHS = 5  # enough after fixing leakage

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Accuracy: {acc:.2f}%")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
torch.save(model.state_dict(), SAVE_PATH)
print(f"\nModel saved to: {SAVE_PATH}")
