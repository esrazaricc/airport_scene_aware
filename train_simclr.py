import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from ssl_dataset import PatchDataset, SimCLRTransform


TRAIN_DIR = r"C:\Users\Esra\Desktop\project\patches\train_final"
SAVE_DIR = r"C:\Users\Esra\Desktop\project\results"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3
TEMPERATURE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = PatchDataset(TRAIN_DIR, transform=SimCLRTransform(image_size=224))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)


class SimCLR(nn.Module):
    def __init__(self, base_model="resnet18", projection_dim=128):
        super().__init__()

        if base_model == "resnet18":
            backbone = models.resnet18(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError("Şimdilik sadece resnet18 destekleniyor.")

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

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim_matrix = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    positives = torch.cat([
        torch.diag(sim_matrix, batch_size),
        torch.diag(sim_matrix, -batch_size)
    ], dim=0)

    denominator = torch.logsumexp(sim_matrix, dim=1)
    loss = -positives + denominator
    return loss.mean()

model = SimCLR().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Cihaz:", DEVICE)
print("Toplam batch:", len(loader))


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for step, (x1, x2) in enumerate(loader):
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)

        _, z1 = model(x1)
        _, z2 = model(x2)

        loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{step+1}/{len(loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Ortalama Loss: {avg_loss:.4f}")

    save_path = os.path.join(SAVE_DIR, f"simclr_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

print("Eğitim tamamlandı.")

