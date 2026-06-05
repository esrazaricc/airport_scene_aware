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
AIRPORT_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_normal"
MODEL_PATH = r"C:\Users\Esra\Desktop\project\results\simclr_epoch_3.pth"
SAVE_FEATURES = r"C:\Users\Esra\Desktop\project\results\airport_train_features.npy"

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
        return image

# =========================
# SimCLR model
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
# Çalıştır
# =========================
dataset = FeatureDataset(AIRPORT_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = SimCLR().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_features = []

print("Cihaz:", DEVICE)
print("Airport patch sayısı:", len(dataset))

with torch.no_grad():
    for step, images in enumerate(loader):
        images = images.to(DEVICE)
        h, _ = model(images)   # 512 boyut backbone feature
        h = h.cpu().numpy()
        all_features.append(h)

        if (step + 1) % 50 == 0:
            print(f"Batch {step+1}/{len(loader)} işlendi")

all_features = np.vstack(all_features)

np.save(SAVE_FEATURES, all_features)

print("Airport feature bank oluşturuldu")
print("Shape:", all_features.shape)
print("Kaydedildi:", SAVE_FEATURES)

