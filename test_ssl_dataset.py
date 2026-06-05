from torch.utils.data import DataLoader
from ssl_dataset import PatchDataset, SimCLRTransform

TRAIN_DIR = r"C:\Users\Esra\Desktop\project\patches\train_final"

dataset = PatchDataset(TRAIN_DIR, transform=SimCLRTransform(image_size=224))
loader = DataLoader(dataset, batch_size=8, shuffle=True)

x1, x2 = next(iter(loader))

print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)

