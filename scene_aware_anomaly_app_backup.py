import os
import textwrap
import numpy as np
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from scipy.ndimage import gaussian_filter

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


# =========================================================
# Patch işlemleri
# =========================================================
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
        return "kuzeybatı"
    if y < y_mid and x >= x_mid:
        return "kuzeydoğu"
    if y >= y_mid and x < x_mid:
        return "güneybatı"

    return "güneydoğu"


def risk_level_from_score(score):
    if score >= 0.80:
        return "Çok Yüksek"
    if score >= 0.60:
        return "Yüksek"
    if score >= 0.30:
        return "Orta"
    return "Düşük"


def scene_specific_explanation(
    semantic_score,
    operational_score,
    non_airport_prob,
    airport_prob,
    dual_error,
    low_error,
    high_error,
    predicted_class,
    region,
    percentile,
    semantic_risk,
    operational_risk
):
    if predicted_class == "non_airport":
        semantic_reason = (
            "Bu patch semantic classifier tarafından non_airport olarak değerlendirilmiştir. "
            "Bu nedenle boundary risk yüksektir."
        )
    else:
        semantic_reason = (
            "Bu patch semantic classifier tarafından airport_component olarak değerlendirilmiştir. "
            "Bu nedenle boundary risk düşüktür."
        )

    text = (
        f"En yüksek semantic boundary risk {region} bölümündedir. "
        f"{semantic_reason}\n\n"
        f"Semantic risk seviyesi: {semantic_risk}. "
        f"Operational anomaly seviyesi: {operational_risk}. "
        f"Semantic risk score: {semantic_score:.4f}. "
        f"Operational anomaly score: {operational_score:.4f}. "
        f"Non-airport probability: {non_airport_prob:.4f}. "
        f"Airport component probability: {airport_prob:.4f}. "
        f"Dual AE error: {dual_error:.6f}. "
        f"Low-level reconstruction error: {low_error:.6f}. "
        f"High-level reconstruction error: {high_error:.6f}. "
        f"Görüntü içi yüzdelik dilim: %{percentile:.1f}. "
        f"Bu sürümde semantic boundary heatmap non_airport_probability ile, "
        f"operational anomaly heatmap ise yalnızca airport_component alanlardaki Dual AutoEncoder reconstruction error ile oluşturulmuştur."
    )

    return text


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

    
    
    # airport dısı alanlar
    
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
    semantic_top_score = float(semantic_scores[semantic_top_idx])
    top_x, top_y = coords[semantic_top_idx]

    operational_top_idx = int(np.argmax(operational_scores))
    operational_top_score = float(operational_scores[operational_top_idx])

    image_h, image_w = full_arr.shape[:2]
    region = region_name_from_coord(top_x, top_y, image_w, image_h)

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

    patch_percentiles = percentile_scores(semantic_scores)
    top_percentile = float(patch_percentiles[semantic_top_idx])

    predicted_class_name = CLASS_NAMES[int(predicted_classes[semantic_top_idx])]
    semantic_risk = risk_level_from_score(semantic_top_score)
    operational_risk = risk_level_from_score(operational_top_score)

    explanation = scene_specific_explanation(
        semantic_score=semantic_top_score,
        operational_score=operational_top_score,
        non_airport_prob=float(non_airport_probs[semantic_top_idx]),
        airport_prob=float(airport_probs[semantic_top_idx]),
        dual_error=float(dual_errors[semantic_top_idx]),
        low_error=float(low_errors[semantic_top_idx]),
        high_error=float(high_errors[semantic_top_idx]),
        predicted_class=predicted_class_name,
        region=region,
        percentile=top_percentile,
        semantic_risk=semantic_risk,
        operational_risk=operational_risk
    )

    semantic_order = np.argsort(semantic_scores)[::-1]
    operational_order = np.argsort(operational_scores)[::-1]

    semantic_top_items = []

    for idx in semantic_order[:5]:
        x, y = coords[idx]
        class_name = CLASS_NAMES[int(predicted_classes[idx])]

        semantic_top_items.append({
            "rank": len(semantic_top_items) + 1,
            "score": float(semantic_scores[idx]),
            "airport_prob": float(airport_probs[idx]),
            "non_airport_prob": float(non_airport_probs[idx]),
            "dual_error": float(dual_errors[idx]),
            "low_error": float(low_errors[idx]),
            "high_error": float(high_errors[idx]),
            "predicted_class": class_name,
            "x": int(x),
            "y": int(y),
            "region": region_name_from_coord(x, y, image_w, image_h),
            "percentile": float(patch_percentiles[idx]),
        })

    operational_top_items = []

    for idx in operational_order[:5]:
        x, y = coords[idx]
        class_name = CLASS_NAMES[int(predicted_classes[idx])]

        operational_top_items.append({
            "rank": len(operational_top_items) + 1,
            "score": float(operational_scores[idx]),
            "dual_error": float(dual_errors[idx]),
            "low_error": float(low_errors[idx]),
            "high_error": float(high_errors[idx]),
            "airport_prob": float(airport_probs[idx]),
            "non_airport_prob": float(non_airport_probs[idx]),
            "predicted_class": class_name,
            "x": int(x),
            "y": int(y),
            "region": region_name_from_coord(x, y, image_w, image_h),
        })

    return {
        "pil_img": pil_img,
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
        "top_region": region,
        "explanation": explanation,

        "semantic_top_items": semantic_top_items,
        "operational_top_items": operational_top_items,

        "patch_count": len(patches),

        "airport_prob_top": float(airport_probs[semantic_top_idx]),
        "non_airport_prob_top": float(non_airport_probs[semantic_top_idx]),
        "dual_error_top": float(dual_errors[semantic_top_idx]),
        "low_error_top": float(low_errors[semantic_top_idx]),
        "high_error_top": float(high_errors[semantic_top_idx]),
        "predicted_class_top": predicted_class_name,
    }

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
    ax.set_title("Orijinal görüntü - en yüksek semantic boundary risk patch")
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
    page_title="Airport Boundary + Operational Risk Demo",
    layout="wide"
)

st.title("Airport Boundary + Operational Risk Demo")
st.caption(
    "Semantic Boundary Risk ve Operational Anomaly Risk ayrı ayrı analiz edilir."
)

left, center, right = st.columns([1, 1.2, 1.2])

with left:
    st.subheader("Kontrol paneli")

    uploaded_file = st.file_uploader(
        "Görüntü yükle",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    run_btn = st.button("Analiz et", type="primary")

    st.markdown("### Model mantığı")
    st.write(
        "1. Semantic Boundary Heatmap = non_airport_probability"
    )
    st.write(
        "2. Operational Anomaly Heatmap = airport_component alanlardaki Dual AE reconstruction error"
    )
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

            with left:
                st.markdown("### Sonuç özeti")
                st.metric("Semantic Global Risk", f"{result['semantic_global_score']:.4f}")
                st.metric("Semantic Risk", result["semantic_risk"])
                st.metric("Top Semantic Patch Score", f"{result['semantic_top_score']:.4f}")

                st.metric("Operational Global Risk", f"{result['operational_global_score']:.4f}")
                st.metric("Operational Risk", result["operational_risk"])
                st.metric("Top Operational Score", f"{result['operational_top_score']:.4f}")

                st.metric("Toplam patch", result["patch_count"])

                st.markdown("### En yüksek semantic risk patch")
                st.write(f"Predicted class: {result['predicted_class_top']}")
                st.write(f"Airport component probability: {result['airport_prob_top']:.4f}")
                st.write(f"Non-airport probability: {result['non_airport_prob_top']:.4f}")

                st.markdown("### Dual AE sinyal")
                st.write(f"Dual AE error: {result['dual_error_top']:.6f}")
                st.write(f"Low-level error: {result['low_error_top']:.6f}")
                st.write(f"High-level error: {result['high_error_top']:.6f}")

                st.markdown("### Top 5 Semantic Boundary Risk")

                for item in result["semantic_top_items"]:
                    st.write(
                        f"{item['rank']}. score={item['score']:.4f} | "
                        f"class={item['predicted_class']} | "
                        f"non_airport={item['non_airport_prob']:.4f} | "
                        f"airport={item['airport_prob']:.4f} | "
                        f"coord=({item['x']}, {item['y']}) | "
                        f"bölge={item['region']}"
                    )

                st.markdown("### Top 5 Operational Anomaly Risk")

                for item in result["operational_top_items"]:
                    st.write(
                        f"{item['rank']}. score={item['score']:.4f} | "
                        f"class={item['predicted_class']} | "
                        f"dual_error={item['dual_error']:.6f} | "
                        f"airport={item['airport_prob']:.4f} | "
                        f"coord=({item['x']}, {item['y']}) | "
                        f"bölge={item['region']}"
                    )

            with center:
                st.subheader("Orijinal görüntü")

                fig_original = draw_original_with_box(
                    result["full_arr"],
                    result["top_coord"],
                    patch_size=PATCH_SIZE
                )

                st.pyplot(fig_original, clear_figure=True)

            with right:
                st.subheader("Semantic Boundary Heatmap")

                fig_semantic = draw_heatmap_overlay(
                    result["full_arr"],
                    result["semantic_heatmap"],
                    title="Semantic Boundary Risk - Non-Airport Probability"
                )

                st.pyplot(fig_semantic, clear_figure=True)

                st.subheader("Operational Anomaly Heatmap")

                fig_operational = draw_heatmap_overlay(
                    result["full_arr"],
                    result["operational_heatmap"],
                    title="Operational Anomaly Risk - Dual AE Error"
                )

                st.pyplot(fig_operational, clear_figure=True)

                st.markdown("### Explanation")
                st.write(result["explanation"])

        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {e}")

else:
    with center:
        st.info(
            "Kullanım: görüntüyü yükle ve Analiz et butonuna bas."
        )

    with right:
        st.markdown(
            textwrap.dedent(
                """
                ### Bu demo ne gösteriyor?
                - Semantic Boundary Heatmap, airport dışı alanları gösterir.
                - Operational Anomaly Heatmap, airport içinde normal örüntüden sapan alanları gösterir.
                - Binary semantic classifier airport_component / non_airport ayrımı yapar.
                - Dual AutoEncoder yalnızca airport_component alanlarda operasyonel sapma sinyali üretir.
                - Böylece soil/vegetation/urban gibi alanlar semantic risk; terminal/apron içi farklı dokular ise operational anomaly olarak ayrı değerlendirilir.
                """
            )
        )
