import os
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image

SOURCE_DIR = r"C:\Users\Esra\Desktop\project\patches\airport_128\train"
OUTPUT_DIR = r"C:\Users\Esra\Desktop\project\urban_candidates"
CSV_PATH = r"C:\Users\Esra\Desktop\project\urban_candidates_scores.csv"

TOP_K = 1500

os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(valid_ext)
]

results = []

def compute_urban_score(img_rgb):
    img = cv2.resize(img_rgb, (128, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

   
    edges = cv2.Canny(gray, 60, 140)
    edge_density = np.mean(edges > 0)

    rgb_std = np.std(img.astype(np.float32), axis=2)
    gray_like_ratio = np.mean(rgb_std < 22)

    vegetation_ratio = np.mean((g > r * 1.12) & (g > b * 1.12) & (g > 45))

    dark_ratio = np.mean(v < 35)

    low_saturation_ratio = np.mean(s < 70)

    # Parlak çatı
    bright_gray_ratio = np.mean((v > 90) & (s < 80))

    # Doku farklılıkları
    texture_score = np.std(gray) / 255
    score = (
        2.5 * edge_density +
        1.5 * gray_like_ratio +
        1.2 * low_saturation_ratio +
        1.2 * bright_gray_ratio +
        1.0 * texture_score -
        2.5 * vegetation_ratio -
        1.5 * dark_ratio
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

print("Toplam dosya:", len(files))

for i, filename in enumerate(files):
    path = os.path.join(SOURCE_DIR, filename)

    try:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        features = compute_urban_score(img_np)
        features["filename"] = filename
        features["path"] = path

        results.append(features)

    except Exception as e:
        print("Okunamadı:", filename, e)

    if (i + 1) % 1000 == 0:
        print(f"İşlenen: {i+1}/{len(files)}")

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
print("Aday klasörü:", OUTPUT_DIR)
print("CSV:", CSV_PATH)
print("Kopyalanan aday sayısı:", len(top_df))
