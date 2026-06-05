import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# =========================
# Yollar
# =========================
TRAIN_DIR = r"C:\Users\Esra\Desktop\project\patches\train_final"
MODEL_PATH = r"C:\Users\Esra\Desktop\project\results\simclr_epoch_3.pth"
SAVE_FEATURES = r"C:\Users\Esra\Desktop\project\results\train_features.npy"
SAVE_PATHS = r"C:\Users\Esra\Desktop\project\results\train_feature_paths.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

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
# Yükleme
# =========================
dataset = FeatureDataset(TRAIN_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = SimCLR().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_features = []
all_paths = []

print("Cihaz:", DEVICE)
print("Toplam görüntü:", len(dataset))

with torch.no_grad():
    for step, (images, paths) in enumerate(loader):
        images = images.to(DEVICE)
        h, _ = model(images)   # 512 boyut backbone feature
        h = h.cpu().numpy()

        all_features.append(h)
        all_paths.extend(paths)

        if (step + 1) % 100 == 0:
            print(f"Batch {step+1}/{len(loader)} işlendi")

all_features = np.vstack(all_features)

np.save(SAVE_FEATURES, all_features)
np.save(SAVE_PATHS, np.array(all_paths))

print("Feature çıkarma tamamlandı")
print("Feature shape:", all_features.shape)
print("Kaydedildi:", SAVE_FEATURES)

