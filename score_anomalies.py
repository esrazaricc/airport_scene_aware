import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.neighbors import NearestNeighbors

# =========================
# Yollar
# =========================
VAL_DIR = r"C:\Users\Esra\Desktop\project\patches\val_normal"
MODEL_PATH = r"C:\Users\Esra\Desktop\project\results\simclr_epoch_3.pth"
TRAIN_FEATURES_PATH = r"C:\Users\Esra\Desktop\project\results\train_features.npy"

SAVE_SCORES = r"C:\Users\Esra\Desktop\project\results\val_anomaly_scores.npy"
SAVE_PATHS = r"C:\Users\Esra\Desktop\project\results\val_paths.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
K = 5

# =========================
# Dataset
# =========================
class FeatureDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, path

# =========================
# Model
# =========================
class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        z = nn.functional.normalize(z, dim=1)
        return h, z

# =========================
# Validation feature çıkarma
# =========================
dataset = FeatureDataset(VAL_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = SimCLR().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

val_features = []
val_paths = []

print("Cihaz:", DEVICE)
print("Val görüntü sayısı:", len(dataset))

with torch.no_grad():
    for step, (images, paths) in enumerate(loader):
        images = images.to(DEVICE)
        h, _ = model(images)
        h = h.cpu().numpy()

        val_features.append(h)
        val_paths.extend(paths)

        if (step + 1) % 50 == 0:
            print(f"Batch {step+1}/{len(loader)} işlendi")

val_features = np.vstack(val_features)

# =========================
# Train feature bank yükle
# =========================
train_features = np.load(TRAIN_FEATURES_PATH)

print("Train feature shape:", train_features.shape)
print("Val feature shape:", val_features.shape)

# =========================
# k-NN anomaly score
# =========================
knn = NearestNeighbors(n_neighbors=K, metric="euclidean")
knn.fit(train_features)

distances, indices = knn.kneighbors(val_features)

# Her örnek için k komşunun ortalama uzaklığı = anomaly score
scores = distances.mean(axis=1)

np.save(SAVE_SCORES, scores)
np.save(SAVE_PATHS, np.array(val_paths))

print("Skorlama tamamlandı")
print("Score shape:", scores.shape)
print("Min score:", scores.min())
print("Max score:", scores.max())
print("Mean score:", scores.mean())


