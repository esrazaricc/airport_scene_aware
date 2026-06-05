import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# =========================
# Ayarlar
# =========================
TRAIN_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_128\train"
SAVE_PATH = r"C:\Users\Esra\Desktop\project\results\airport_high_ae.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-4
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
# High-level AE (kernel=5)
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
dataset = PatchDataset(TRAIN_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# Model
# =========================
model = HighLevelAE().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Cihaz:", DEVICE)
print("Train patch:", len(dataset))

# =========================
# Training
# =========================
for epoch in range(EPOCHS):
    total_loss = 0

    for step, images in enumerate(loader):
        images = images.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step+1) % 50 == 0:
            print(f"Epoch {epoch+1} Step {step+1} Loss {loss.item():.6f}")

    print(f"Epoch {epoch+1} Avg Loss {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), SAVE_PATH)
print("High-level AE kaydedildi:", SAVE_PATH)

