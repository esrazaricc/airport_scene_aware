import os
import cv2
import numpy as np

RAW_DIR = r"C:\Users\Esra\Desktop\project\raw_data"
CLEAN_DIR = r"C:\Users\Esra\Desktop\project\cleaned_data"

os.makedirs(CLEAN_DIR, exist_ok=True)

def is_too_dark(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < 10

def is_too_bright(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > 245

def is_low_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray) < 10

def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score < 40

kept = 0
removed = 0

for file in os.listdir(RAW_DIR):

    if not file.lower().endswith(".tif"):
        continue

    path = os.path.join(RAW_DIR, file)
    img = cv2.imread(path)

    if img is None:
        removed += 1
        continue

    if is_too_dark(img):
        removed += 1
        continue

    if is_too_bright(img):
        removed += 1
        continue

    if is_low_variance(img):
        removed += 1
        continue

    if is_blurry(img):
        removed += 1
        continue

    save_path = os.path.join(CLEAN_DIR, file)
    cv2.imwrite(save_path, img)
    kept += 1

print("Temizleme tamamlandı")
print("Tutulan görüntü:", kept)
print("Silinen görüntü:", removed)

