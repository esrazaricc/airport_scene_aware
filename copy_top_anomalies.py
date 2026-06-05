import numpy as np
import shutil
import os

SCORES_PATH = r"C:\Users\Esra\Desktop\project\results\val_anomaly_scores.npy"
PATHS_PATH = r"C:\Users\Esra\Desktop\project\results\val_paths.npy"

OUTPUT_DIR = r"C:\Users\Esra\Desktop\project\results\top_anomalies"

TOP_K = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

scores = np.load(SCORES_PATH)
paths = np.load(PATHS_PATH, allow_pickle=True)

sorted_idx = np.argsort(scores)[::-1]

for i in range(TOP_K):
    idx = sorted_idx[i]
    src = paths[idx]
    
    filename = f"{i+1}_score_{scores[idx]:.2f}.png"
    dst = os.path.join(OUTPUT_DIR, filename)
    
    shutil.copy(src, dst)

print(f"{TOP_K} tane en yüksek anomali patch'i kopyalandı.")
print("Klasör:", OUTPUT_DIR)


