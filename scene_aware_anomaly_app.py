import os
import textwrap
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

PROJECT_ROOT = r"C:\Users\Esra\Desktop\project"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

LOW_AE_PATH = os.path.join(RESULTS_DIR, "airport_low_ae.pth")
HIGH_AE_PATH = os.path.join(RESULTS_DIR, "airport_high_ae.pth")
BINARY_CLASSIFIER_PATH = os.path.join(RESULTS_DIR, "binary_semantic_classifier.pth")

PATCH_SIZE = 128
STRIDE = 64
IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["airport_component", "non_airport"]

class LowLevelAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class HighLevelAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))



@st.cache_resource
def load_dual_ae_models():
    low_model = LowLevelAE().to(DEVICE)
    high_model = HighLevelAE().to(DEVICE)

    low_model.load_state_dict(torch.load(LOW_AE_PATH, map_location=DEVICE))
    high_model.load_state_dict(torch.load(HIGH_AE_PATH, map_location=DEVICE))

    low_model.eval()
    high_model.eval()

    return low_model, high_model


@st.cache_resource
def load_binary_classifier():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(BINARY_CLASSIFIER_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    model = model.to(DEVICE)
    model.eval()

    return model


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
#patch işlemi
def image_to_patches(pil_img: Image.Image, patch_size=128, stride=64):
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    h, w = arr.shape[:2]

    patches = []
    coords = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = arr[y:y + patch_size, x:x + patch_size]
            patches.append(Image.fromarray(patch))
            coords.append((x, y))

    return patches, coords, arr


def patches_to_tensor(patches):
    tensors = [transform(p) for p in patches]
    return torch.stack(tensors).to(DEVICE)



# Binary SC
def score_patches_binary_classifier(patches, batch_size=64):
    model = load_binary_classifier()

    airport_probs = []
    non_airport_probs = []
    predicted_classes = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            x = patches_to_tensor(batch_patches)

            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            airport_prob = probs[:, 0]
            non_airport_prob = probs[:, 1]
            preds = torch.argmax(probs, dim=1)

            airport_probs.extend(airport_prob.cpu().numpy())
            non_airport_probs.extend(non_airport_prob.cpu().numpy())
            predicted_classes.extend(preds.cpu().numpy())

    return (
        np.array(airport_probs, dtype=np.float32),
        np.array(non_airport_probs, dtype=np.float32),
        np.array(predicted_classes, dtype=np.int64)
    )



def score_patches_dual_ae(patches, batch_size=64):
    low_model, high_model = load_dual_ae_models()

    low_errors = []
    high_errors = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            x = patches_to_tensor(batch_patches)

            low_out = low_model(x)
            high_out = high_model(x)

            low_err = torch.mean((low_out - x) ** 2, dim=(1, 2, 3))
            high_err = torch.mean((high_out - x) ** 2, dim=(1, 2, 3))

            low_errors.extend(low_err.cpu().numpy())
            high_errors.extend(high_err.cpu().numpy())

    low_errors = np.array(low_errors, dtype=np.float32)
    high_errors = np.array(high_errors, dtype=np.float32)
    dual_errors = low_errors + high_errors

    return dual_errors, low_errors, high_errors

def normalize_scores(scores):
    mn = float(scores.min())
    mx = float(scores.max())

    if mx - mn < 1e-8:
        return np.zeros_like(scores)

    return (scores - mn) / (mx - mn)


def percentile_scores(scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    percentiles = 100.0 * (ranks + 1) / len(scores)
    return percentiles


def build_heatmap_canvas(image_shape, coords, scores, patch_size=128):
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for (x, y), s in zip(coords, scores):
        heatmap[y:y + patch_size, x:x + patch_size] += float(s)
        counts[y:y + patch_size, x:x + patch_size] += 1.0

    counts[counts == 0] = 1.0
    heatmap = heatmap / counts

    heatmap = gaussian_filter(heatmap, sigma=10)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap


def region_name_from_coord(x, y, w, h):
    x_mid = w / 2
    y_mid = h / 2

    if y < y_mid and x < x_mid:
        return "Kuzeybatı"
    if y < y_mid and x >= x_mid:
        return "Kuzeydoğu"
    if y >= y_mid and x < x_mid:
        return "Güneybatı"

    return "Güneydoğu"


def risk_level_from_score(score):
    if score >= 0.80:
        return "Çok Yüksek"
    if score >= 0.60:
        return "Yüksek"
    if score >= 0.30:
        return "Orta"
    return "Düşük"


def explanation_text(result):
    return (
        f"Bu analizde iki farklı risk katmanı ayrı ayrı değerlendirilmiştir. "
        f"Semantic Boundary Heatmap, airport dışı alanları tespit etmek için "
        f"non_airport_probability değerini kullanır. Operational Anomaly Heatmap ise "
        f"yalnızca airport_component olarak değerlendirilen bölgelerde Dual AutoEncoder "
        f"reconstruction error değerini kullanır.\n\n"
        f"En yüksek semantic risk bölgesi: {result['top_region']}. "
        f"Semantic risk skoru: {result['semantic_top_score']:.4f}. "
        f"Semantic risk seviyesi: {result['semantic_risk']}. "
        f"Operational risk skoru: {result['operational_top_score']:.4f}. "
        f"Operational risk seviyesi: {result['operational_risk']}."
    )

def analyze_uploaded_image(uploaded_file):
    pil_img = Image.open(uploaded_file).convert("RGB")

    patches, coords, full_arr = image_to_patches(
        pil_img,
        patch_size=PATCH_SIZE,
        stride=STRIDE
    )

    if len(patches) == 0:
        raise ValueError(
            f"Görüntü en az {PATCH_SIZE}x{PATCH_SIZE} olmalı. Daha büyük bir görüntü yükle."
        )

    airport_probs, non_airport_probs, predicted_classes = score_patches_binary_classifier(patches)

    dual_errors, low_errors, high_errors = score_patches_dual_ae(patches)
    dual_errors_norm = normalize_scores(dual_errors)

    
    semantic_scores = non_airport_probs.copy()

    airport_confident_mask = airport_probs > 0.70
    semantic_scores[airport_confident_mask] *= 0.05

    non_airport_confident_mask = non_airport_probs > 0.70
    semantic_scores[non_airport_confident_mask] = np.maximum(
        semantic_scores[non_airport_confident_mask],
        0.85
    )

    semantic_scores[semantic_scores < 0.20] = 0.0
    semantic_scores = np.clip(semantic_scores, 0.0, 1.0)

    operational_scores = dual_errors_norm.copy()
    operational_scores[non_airport_probs > 0.70] = 0.0
    operational_scores[airport_probs < 0.50] = 0.0
    operational_scores[operational_scores < 0.20] = 0.0
    operational_scores = np.clip(operational_scores, 0.0, 1.0)

    semantic_top_idx = int(np.argmax(semantic_scores))
    operational_top_idx = int(np.argmax(operational_scores))

    semantic_top_score = float(semantic_scores[semantic_top_idx])
    operational_top_score = float(operational_scores[operational_top_idx])

    top_x, top_y = coords[semantic_top_idx]
    image_h, image_w = full_arr.shape[:2]
    top_region = region_name_from_coord(top_x, top_y, image_w, image_h)

    semantic_heatmap = build_heatmap_canvas(
        full_arr.shape,
        coords,
        semantic_scores,
        patch_size=PATCH_SIZE
    )

    operational_heatmap = build_heatmap_canvas(
        full_arr.shape,
        coords,
        operational_scores,
        patch_size=PATCH_SIZE
    )

    semantic_global_score = float(
        np.mean(
            np.sort(semantic_scores)[-max(1, len(semantic_scores) // 10):]
        )
    )

    operational_global_score = float(
        np.mean(
            np.sort(operational_scores)[-max(1, len(operational_scores) // 10):]
        )
    )

    semantic_risk = risk_level_from_score(semantic_top_score)
    operational_risk = risk_level_from_score(operational_top_score)

    patch_percentiles = percentile_scores(semantic_scores)

    semantic_order = np.argsort(semantic_scores)[::-1]
    operational_order = np.argsort(operational_scores)[::-1]

    semantic_top_items = []
    for idx in semantic_order[:5]:
        x, y = coords[idx]
        semantic_top_items.append({
            "#": len(semantic_top_items) + 1,
            "Bölge": region_name_from_coord(x, y, image_w, image_h),
            "Sınıf": CLASS_NAMES[int(predicted_classes[idx])],
            "Semantic Risk": round(float(semantic_scores[idx]), 4),
            "Non-Airport Prob.": round(float(non_airport_probs[idx]), 4),
            "Airport Prob.": round(float(airport_probs[idx]), 4),
            "Dual AE Error": round(float(dual_errors[idx]), 6),
            "Koordinat": f"({int(x)}, {int(y)})"
        })

    operational_top_items = []
    for idx in operational_order[:5]:
        x, y = coords[idx]
        operational_top_items.append({
            "#": len(operational_top_items) + 1,
            "Bölge": region_name_from_coord(x, y, image_w, image_h),
            "Sınıf": CLASS_NAMES[int(predicted_classes[idx])],
            "Operational Risk": round(float(operational_scores[idx]), 4),
            "Dual AE Error": round(float(dual_errors[idx]), 6),
            "Low AE Error": round(float(low_errors[idx]), 6),
            "High AE Error": round(float(high_errors[idx]), 6),
            "Airport Prob.": round(float(airport_probs[idx]), 4),
            "Koordinat": f"({int(x)}, {int(y)})"
        })

    return {
        "full_arr": full_arr,

        "semantic_heatmap": semantic_heatmap,
        "operational_heatmap": operational_heatmap,

        "semantic_global_score": semantic_global_score,
        "operational_global_score": operational_global_score,

        "semantic_risk": semantic_risk,
        "operational_risk": operational_risk,

        "semantic_top_score": semantic_top_score,
        "operational_top_score": operational_top_score,

        "top_coord": (top_x, top_y),
        "top_region": top_region,

        "patch_count": len(patches),

        "semantic_top_items": semantic_top_items,
        "operational_top_items": operational_top_items,

        "predicted_class_top": CLASS_NAMES[int(predicted_classes[semantic_top_idx])],
        "airport_prob_top": float(airport_probs[semantic_top_idx]),
        "non_airport_prob_top": float(non_airport_probs[semantic_top_idx]),
        "dual_error_top": float(dual_errors[semantic_top_idx]),
        "low_error_top": float(low_errors[semantic_top_idx]),
        "high_error_top": float(high_errors[semantic_top_idx]),
        "percentile_top": float(patch_percentiles[semantic_top_idx]),
    }

# Görselleştirme

def draw_original_with_box(image_arr, top_coord, patch_size=128):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_arr)

    x, y = top_coord

    rect = Rectangle(
        (x, y),
        patch_size,
        patch_size,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rect)
    ax.set_title("Orijinal Görüntü - En Riskli Semantic Patch")
    ax.axis("off")

    return fig


def draw_heatmap_overlay(image_arr, heatmap, title="Heatmap"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_arr)
    ax.imshow(
        heatmap,
        alpha=0.50,
        cmap="jet",
        vmin=0.0,
        vmax=1.0
    )
    ax.set_title(title)
    ax.axis("off")

    return fig

st.set_page_config(
    page_title="Havalimanı Semantic Risk Analizi",
    layout="wide"
)

st.title("Havalimanı Semantic Risk Analizi")
st.caption(
    "Semantic Boundary Heatmap airport dışı alanları, "
    "Operational Anomaly Heatmap ise airport içindeki yapısal sapmaları gösterir."
)

left_panel, main_panel = st.columns([1, 3])

with left_panel:
    st.subheader("Kontrol Paneli")

    uploaded_file = st.file_uploader(
        "Görüntü yükle",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    run_btn = st.button("Analiz Et", type="primary")

    st.markdown("---")

    st.markdown("### Model Mantığı")
    st.write("**1. Semantic Boundary Heatmap**")
    st.write("non_airport_probability ile oluşturulur.")

    st.write("**2. Operational Anomaly Heatmap**")
    st.write("Airport_component alanlardaki Dual AE reconstruction error ile oluşturulur.")

    st.info(
        "Airport component: apron, runway, terminal, aircraft. "
        "Non-airport: soil, vegetation, urban, water veya airport dışı çevresel alan."
    )


if run_btn:
    if uploaded_file is None:
        st.error("Önce bir görüntü yükle.")
    else:
        try:
            result = analyze_uploaded_image(uploaded_file)

            with left_panel:
                st.markdown("### Sonuç Özeti")

                st.metric("Semantic Global Risk", f"{result['semantic_global_score']:.4f}")
                st.metric("Semantic Risk", result["semantic_risk"])
                st.metric("Top Semantic Patch Score", f"{result['semantic_top_score']:.4f}")

                st.metric("Operational Global Risk", f"{result['operational_global_score']:.4f}")
                st.metric("Operational Risk", result["operational_risk"])
                st.metric("Top Operational Score", f"{result['operational_top_score']:.4f}")

                st.metric("Toplam Patch", result["patch_count"])

                st.markdown("### En Riskli Patch")
                st.write(f"Predicted class: `{result['predicted_class_top']}`")
                st.write(f"Airport prob: `{result['airport_prob_top']:.4f}`")
                st.write(f"Non-airport prob: `{result['non_airport_prob_top']:.4f}`")
                st.write(f"Dual AE error: `{result['dual_error_top']:.6f}`")

            with main_panel:
                top_left, top_right = st.columns(2)

                with top_left:
                    st.subheader("Orijinal Görüntü")
                    fig_original = draw_original_with_box(
                        result["full_arr"],
                        result["top_coord"],
                        patch_size=PATCH_SIZE
                    )
                    st.pyplot(fig_original, clear_figure=True)

                with top_right:
                    st.subheader("Semantic Boundary Heatmap")
                    fig_semantic = draw_heatmap_overlay(
                        result["full_arr"],
                        result["semantic_heatmap"],
                        title="Semantic Boundary Risk - Non-Airport Probability"
                    )
                    st.pyplot(fig_semantic, clear_figure=True)

                st.markdown("---")

                bottom_left, bottom_right = st.columns(2)

                with bottom_left:
                    st.subheader("Operational Anomaly Heatmap")
                    fig_operational = draw_heatmap_overlay(
                        result["full_arr"],
                        result["operational_heatmap"],
                        title="Operational Anomaly Risk - Dual AE Error"
                    )
                    st.pyplot(fig_operational, clear_figure=True)

                with bottom_right:
                    st.subheader("En Riskli Patch Detayı")

                    detail_df = pd.DataFrame({
                        "Alan": [
                            "Bölge",
                            "Tahmin Edilen Sınıf",
                            "Semantic Risk Score",
                            "Airport Probability",
                            "Non-Airport Probability",
                            "Dual AE Error",
                            "Low AE Error",
                            "High AE Error",
                            "Yüzdelik Dilim"
                        ],
                        "Değer": [
                            result["top_region"],
                            result["predicted_class_top"],
                            f"{result['semantic_top_score']:.4f}",
                            f"{result['airport_prob_top']:.4f}",
                            f"{result['non_airport_prob_top']:.4f}",
                            f"{result['dual_error_top']:.6f}",
                            f"{result['low_error_top']:.6f}",
                            f"{result['high_error_top']:.6f}",
                            f"%{result['percentile_top']:.1f}"
                        ]
                    })

                    st.table(detail_df)

                st.markdown("---")

                table_left, table_right = st.columns(2)

                with table_left:
                    st.subheader("Top 5 Semantic Boundary Risk")
                    st.dataframe(
                        pd.DataFrame(result["semantic_top_items"]),
                        use_container_width=True
                    )

                with table_right:
                    st.subheader("Top 5 Operational Anomaly Risk")
                    st.dataframe(
                        pd.DataFrame(result["operational_top_items"]),
                        use_container_width=True
                    )

                st.markdown("---")

                st.subheader("Açıklama")
                st.write(explanation_text(result))

        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {e}")

else:
    with main_panel:
        st.info("Kullanım: Görüntüyü yükle ve Analiz Et butonuna bas.")

        st.markdown(
            textwrap.dedent(
                """
                ### Sistem ne yapar?

                Bu uygulama havalimanı görüntülerini iki farklı risk katmanında analiz eder:

                **1. Semantic Boundary Risk**  
                Airport dışı alanları tespit eder. Soil, vegetation, urban ve water gibi alanlar riskli kabul edilir.

                **2. Operational Anomaly Risk**  
                Airport içinde normal örüntüden sapan bölgeleri tespit eder. Bu katmanda Dual AutoEncoder reconstruction error kullanılır.

                Bu nedenle sistem yalnızca tek bir anomaly skoru üretmez; boundary riski ve operasyonel anomaly riski ayrı ayrı raporlar.
                """
            )
        )
