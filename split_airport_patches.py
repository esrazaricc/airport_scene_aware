import os
import random
import shutil

SOURCE_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_128_all"
TRAIN_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_128\train"
VAL_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_128\val"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
]

random.seed(42)
random.shuffle(files)

split_idx = int(len(files) * 0.8)

train_files = files[:split_idx]
val_files = files[split_idx:]

for f in train_files:
    shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(TRAIN_DIR, f))

for f in val_files:
    shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(VAL_DIR, f))

print("Train:", len(train_files))
print("Val:", len(val_files))