import os
import random
import shutil

SOURCE_DIR = r"C:\Users\Esra\Desktop\project\patches\train_normal"
TRAIN_DIR = r"C:\Users\Esra\Desktop\project\patches\train_final"
VAL_DIR = r"C:\Users\Esra\Desktop\project\patches\val_normal"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".png")]
random.shuffle(files)

n = len(files)
train_end = int(n * 0.85)

train_files = files[:train_end]
val_files = files[train_end:]

for f in train_files:
    shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(TRAIN_DIR, f))

for f in val_files:
    shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(VAL_DIR, f))

print("Bölme tamamlandı")
print("Train:", len(train_files))
print("Val:", len(val_files))

