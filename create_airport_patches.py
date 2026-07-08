import os
import cv2
import numpy as np

INPUT_DIR = r"C:\Users\Esra\Desktop\project\airport_images"
OUTPUT_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_128_all"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 128
STRIDE = 128

def patch_is_useful(patch, dark_thresh=10, std_thresh=12):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

   
    if np.mean(gray) < dark_thresh:
        return False

    # Çok düz
    if np.std(gray) < std_thresh:
        return False

    return True

saved_count = 0
skipped_count = 0

for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(path)

    if img is None:
        print(f"Okunamadı: {file}")
        continue

    h, w = img.shape[:2]
    print(f"{file} -> boyut: {w}x{h}")

    if h < PATCH_SIZE or w < PATCH_SIZE:
        print(f"PATCH_SIZE büyük olduğu için patch çıkmaz: {file}")
        continue

    file_saved = 0
    file_skipped = 0

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):

            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                skipped_count += 1
                file_skipped += 1
                continue

            if not patch_is_useful(patch):
                skipped_count += 1
                file_skipped += 1
                continue

            name = f"{os.path.splitext(file)[0]}_y{y}_x{x}.png"
            out_path = os.path.join(OUTPUT_DIR, name)

            cv2.imwrite(out_path, patch)
            saved_count += 1
            file_saved += 1

    print(f"{file} -> kaydedilen: {file_saved}, atlanan: {file_skipped}")

print("Airport patch üretimi tamamlandı")
print("Kaydedilen patch:", saved_count)
print("Atlanan patch:", skipped_count)

