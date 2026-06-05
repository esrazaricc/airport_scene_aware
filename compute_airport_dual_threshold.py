import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# =========================
# Ayarlar
# =========================
VAL_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_128\val"
LOW_MODEL_PATH = r"C:\Users\Esra\Desktop\project\results\airport_low_ae.pth"
HIGH_MODEL_PATH = r"C:\Users\Esra\Desktop\project\results\airport_high_ae.pth"
SAVE_STATS_PATH = r"C:\Users\Esra\Desktop\project\results\airport_dual_threshold_stats.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
IMAGE_SIZE = 128

# =========================
# Dataset
# =========================
class PatchDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# =========================
# Low AE
# =========================
class LowLevelAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =========================
# High AE
# =========================
class HighLevelAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =========================
# Data
# =========================
dataset = PatchDataset(VAL_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================
# Models
# =========================
low_model = LowLevelAE().to(DEVICE)
high_model = HighLevelAE().to(DEVICE)

low_model.load_state_dict(torch.load(LOW_MODEL_PATH, map_location=DEVICE))
high_model.load_state_dict(torch.load(HIGH_MODEL_PATH, map_location=DEVICE))

low_model.eval()
high_model.eval()

print("Cihaz:", DEVICE)
print("Val patch sayısı:", len(dataset))

all_scores = []
all_low = []
all_high = []

with torch.no_grad():
    for step, images in enumerate(loader):
        images = images.to(DEVICE)

        low_out = low_model(images)
        high_out = high_model(images)

        low_err = ((images - low_out) ** 2).mean(dim=(1, 2, 3))
        high_err = ((images - high_out) ** 2).mean(dim=(1, 2, 3))

        total_score = low_err + high_err

        all_low.extend(low_err.cpu().numpy().tolist())
        all_high.extend(high_err.cpu().numpy().tolist())
        all_scores.extend(total_score.cpu().numpy().tolist())

        if (step + 1) % 50 == 0:
            print(f"Batch {step+1}/{len(loader)} işlendi")

all_low = np.array(all_low, dtype=np.float32)
all_high = np.array(all_high, dtype=np.float32)
all_scores = np.array(all_scores, dtype=np.float32)

threshold_mean = float(all_scores.mean())
threshold_std = float(all_scores.std())
threshold_p95 = float(np.percentile(all_scores, 95))

np.savez(
    SAVE_STATS_PATH,
    low_scores=all_low,
    high_scores=all_high,
    total_scores=all_scores,
    threshold_mean=threshold_mean,
    threshold_std=threshold_std,
    threshold_p95=threshold_p95
)

print("Threshold stats kaydedildi:", SAVE_STATS_PATH)
print("Mean threshold:", threshold_mean)
print("Std:", threshold_std)
print("P95 threshold:", threshold_p95)
print("Min:", float(all_scores.min()))
print("Max:", float(all_scores.max()))

