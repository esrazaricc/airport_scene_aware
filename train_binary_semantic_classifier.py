import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score
from PIL import Image
import os
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 128
BATCH_SIZE = 32

# Dataset
class BinaryDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


train_dir_air = r"C:\Users\Esra\Desktop\project\semantic_binary_dataset\train\airport_component"
train_dir_non = r"C:\Users\Esra\Desktop\project\semantic_binary_dataset\train\non_airport"

train_image_paths = [os.path.join(train_dir_air, f) for f in os.listdir(train_dir_air)] + \
                    [os.path.join(train_dir_non, f) for f in os.listdir(train_dir_non)]

train_labels = [1]*len(os.listdir(train_dir_air)) + [0]*len(os.listdir(train_dir_non))

# DataLoader
train_dataset = BinaryDataset(train_image_paths, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 2)  # binary
model = model.to(DEVICE)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Metrikler
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
iou = jaccard_score(all_labels, all_preds)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"IoU: {iou:.4f}")
