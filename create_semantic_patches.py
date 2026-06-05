import os
from PIL import Image

# =====================================================
# Ayarlar
# =====================================================
PROJECT_ROOT = r"C:\Users\Esra\Desktop\project"

INPUT_IMAGE_PATH = r"C:\Users\Esra\Desktop\project\raw_data\1076.tif"

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "semantic_dataset")

PATCH_SIZE = 128
STRIDE = 128

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

SPLIT = "train"  # train veya val

# =====================================================
# Kayıt klasörlerini kontrol et
# =====================================================
for class_name in CLASSES:
    folder = os.path.join(OUTPUT_ROOT, SPLIT, class_name)
    os.makedirs(folder, exist_ok=True)

# =====================================================
# Görüntüyü aç
# =====================================================
img = Image.open(INPUT_IMAGE_PATH).convert("RGB")
w, h = img.size

print("Görüntü boyutu:", w, h)
print("Patch size:", PATCH_SIZE)
print("Split:", SPLIT)
print("Sınıflar:")
for i, c in enumerate(CLASSES):
    print(f"{i}: {c}")

patch_id = 0
saved_count = 0

# =====================================================
# Patch üret ve kullanıcıdan sınıf iste
# =====================================================
for y in range(0, h - PATCH_SIZE + 1, STRIDE):
    for x in range(0, w - PATCH_SIZE + 1, STRIDE):
        patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

        preview_path = os.path.join(PROJECT_ROOT, "current_patch_preview.png")
        patch.save(preview_path)

        print("\n-----------------------------------")
        print(f"Patch ID: {patch_id}")
        print(f"Koordinat: x={x}, y={y}")
        print(f"Önizleme dosyası: {preview_path}")
        print("Sınıf seç:")
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
            print("Çıkılıyor...")
            print("Kaydedilen patch sayısı:", saved_count)
            exit()

        if choice == "s":
            print("Atlandı.")
            patch_id += 1
            continue

        if not choice.isdigit():
            print("Geçersiz seçim, atlandı.")
            patch_id += 1
            continue

        class_index = int(choice)

        if class_index < 0 or class_index >= len(CLASSES):
            print("Geçersiz sınıf, atlandı.")
            patch_id += 1
            continue

        class_name = CLASSES[class_index]

        save_name = f"{class_name}_{patch_id}_x{x}_y{y}.png"
        save_path = os.path.join(OUTPUT_ROOT, SPLIT, class_name, save_name)

        patch.save(save_path)
        saved_count += 1

        print("Kaydedildi:", save_path)

        patch_id += 1

print("Bitti.")
print("Toplam kaydedilen patch:", saved_count)

