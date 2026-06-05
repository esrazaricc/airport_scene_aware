import os
import shutil
from PIL import Image

PROJECT_ROOT = r"C:\Users\Esra\Desktop\project"

SOURCE_PATCH_DIR = os.path.join(PROJECT_ROOT, "airport_roi_images")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "semantic_dataset", "train")
PREVIEW_PATH = os.path.join(PROJECT_ROOT, "current_patch_preview.png")

CLASSES = [
    "aircraft",
    "runway",
    "apron",
    "terminal",
    "urban",
    "soil",
    "vegetation",
    "water"
]

image_files = [
    f for f in os.listdir(SOURCE_PATCH_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
]

print("Toplam patch:", len(image_files))
print("Sınıflar:")
for i, c in enumerate(CLASSES):
    print(f"{i}: {c}")

for idx, filename in enumerate(image_files):
    src_path = os.path.join(SOURCE_PATCH_DIR, filename)

    try:
        img = Image.open(src_path).convert("RGB")
        img.save(PREVIEW_PATH)
    except Exception as e:
        print("Okunamadı:", src_path, e)
        continue

    print("\n--------------------------------")
    print(f"Patch {idx + 1}/{len(image_files)}")
    print("Dosya:", filename)
    print("Önizleme:", PREVIEW_PATH)
    print("0 aircraft")
    print("1 runway")
    print("2 apron")
    print("3 terminal")
    print("4 urban")
    print("5 soil")
    print("6 vegetation")
    print("7 water")
    print("s skip")
    print("q quit")

    choice = input("Seçim: ").strip().lower()

    if choice == "q":
        print("Çıkılıyor.")
        break

    if choice == "s":
        print("Atlandı.")
        continue

    if not choice.isdigit():
        print("Geçersiz seçim, atlandı.")
        continue

    class_index = int(choice)

    if class_index < 0 or class_index >= len(CLASSES):
        print("Geçersiz sınıf, atlandı.")
        continue

    class_name = CLASSES[class_index]
    target_dir = os.path.join(OUTPUT_ROOT, class_name)
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path):
        name, ext = os.path.splitext(filename)
        target_path = os.path.join(target_dir, f"{name}_{idx}{ext}")

    shutil.copy2(src_path, target_path)
    print("Kaydedildi:", target_path)

print("Bitti.")

