import numpy as np
import os

SCORES_PATH = r"C:\Users\Esra\Desktop\project\results\val_anomaly_scores.npy"
PATHS_PATH = r"C:\Users\Esra\Desktop\project\results\val_paths.npy"

TOP_K = 20

scores = np.load(SCORES_PATH)
paths = np.load(PATHS_PATH, allow_pickle=True)

sorted_idx = np.argsort(scores)[::-1]   # büyükten küçüğe

print(f"Top {TOP_K} en yüksek anomali skorlu patch:\n")

for rank, idx in enumerate(sorted_idx[:TOP_K], start=1):
    print(f"{rank}. Score = {scores[idx]:.4f}")
    print(f"   Path  = {paths[idx]}")
    print()


