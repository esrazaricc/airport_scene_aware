import os
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image

SOURCE_DIR = r"C:\Users\Esra\Desktop\project\raw_data"
PATCH_DIR = r"C:\Users\Esra\Desktop\project\raw_urban_patches"
OUTPUT_DIR = r"C:\Users\Esra\Desktop\project\urban_candidates"
CSV_PATH = r"C:\Users\Esra\Desktop\project\urban_candidates_scores.csv"

PATCH_SIZE = 128
STRIDE = 128
TOP_K = 2000

os.makedirs(PATCH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

image_files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(valid_ext)
]

print("Raw image sayısı:", len(image_files))


def compute_urban_score(img_rgb):
    img = cv2.resize(img_rgb, (128, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    edges = cv2.Canny(gray, 60, 140)
    edge_density = np.mean(edges > 0)

    rgb_std = np.std(img.astype(np.float32), axis=2)
    gray_like_ratio = np.mean(rgb_std < 24)

    vegetation_ratio = np.mean((g > r * 1.12) & (g > b * 1.12) & (g > 45))

    dark_ratio = np.mean(v < 35)

    low_saturation_ratio = np.mean(s < 80)

    bright_gray_ratio = np.mean((v > 90) & (s < 90))

    texture_score = np.std(gray) / 255.0

    # şehir/bina/yol skoru
    score = (
        3.0 * edge_density +
        1.7 * gray_like_ratio +
        1.5 * low_saturation_ratio +
        1.3 * bright_gray_ratio +
        1.2 * texture_score -
        2.0 * vegetation_ratio -
        1.2 * dark_ratio
    )

    return {
        "score": float(score),
        "edge_density": float(edge_density),
        "gray_like_ratio": float(gray_like_ratio),
        "vegetation_ratio": float(vegetation_ratio),
        "dark_ratio": float(dark_ratio),
        "low_saturation_ratio": float(low_saturation_ratio),
        "bright_gray_ratio": float(bright_gray_ratio),
        "texture_score": float(texture_score),
    }


results = []
patch_counter = 0

for img_idx, filename in enumerate(image_files, start=1):
    img_path = os.path.join(SOURCE_DIR, filename)

    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
    except Exception as e:
        print("Okunamadı:", filename, e)
        continue

    h, w = arr.shape[:2]
    print(f"[{img_idx}/{len(image_files)}] {filename} | boyut: {w}x{h}")

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = arr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            features = compute_urban_score(patch)

            base_name = os.path.splitext(filename)[0]
            patch_name = f"{base_name}_y{y}_x{x}.png"
            patch_path = os.path.join(PATCH_DIR, patch_name)

            Image.fromarray(patch).save(patch_path)

            features["filename"] = patch_name
            features["path"] = patch_path
            features["source_image"] = filename
            features["x"] = x
            features["y"] = y

            results.append(features)
            patch_counter += 1

print("Toplam çıkarılan patch:", patch_counter)

df = pd.DataFrame(results)
df = df.sort_values("score", ascending=False)
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

top_df = df.head(TOP_K)

for rank, row in enumerate(top_df.itertuples(), start=1):
    src = row.path
    name, ext = os.path.splitext(row.filename)

    dst_name = f"{rank:04d}_score_{row.score:.3f}_{name}{ext}"
    dst = os.path.join(OUTPUT_DIR, dst_name)

    shutil.copy2(src, dst)

print("Bitti.")
print("Patch klasörü:", PATCH_DIR)
print("Urban aday klasörü:", OUTPUT_DIR)
print("CSV:", CSV_PATH)
print("Kopyalanan aday:", len(top_df))
