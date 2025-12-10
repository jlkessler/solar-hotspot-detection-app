import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torchvision.ops import nms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load Hotspot Detection Model
# -------------------------
@st.cache_resource
def load_hotspot_model():
    device = torch.device("cpu")  # or "cuda" if you want GPU and have it set up

    num_classes = 2  # background + hotspot

    # This should match how you built the model during training
    model = fasterrcnn_resnet50_fpn(weights=None)  # or weights="DEFAULT" if you want base pretrained weights too

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load("models/hotspot_detector.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# -------------------------
# ConvLSTM Classes & Utilities
# -------------------------
from utils.convlstm import ConvLSTMModel

@st.cache_resource
def load_convlstm():
    model = ConvLSTMModel(hidden=32)
    model.load_state_dict(torch.load("models/weighted_convlstm_model.pth", map_location="cpu"))
    model.eval()
    return model

def load_grid(csv_file):
    df = pd.read_csv(csv_file, header=None)
    arr = df.values.astype("float32")
    mask = (arr != -1).astype("float32")
    arr[arr == -1] = 0
    return torch.tensor(arr).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

# -------------------------
# Streamlit App Layout
# -------------------------
st.title("🔆 Solar Array Hotspot Detection & Anomaly Forecasting")
st.write("A demonstration of computer vision + time-series modeling for solar panel analytics.")

tabs = st.tabs(["🔥 Hotspot Detection", "📈 Anomaly Forecasting"])

# ======================================================
# 🔥 TAB 1 — OBJECT DETECTION
# ======================================================
with tabs[0]:
    st.header("Hotspot Detection on Thermal Images")

    uploaded = st.file_uploader("Upload a thermal JPG image", type=["jpg", "jpeg", "png"])

    if uploaded:
        model = load_hotspot_model()
        img = Image.open(uploaded).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        x = transform(img)

        with torch.no_grad():
            preds = model([x])[0]

        boxes = preds["boxes"]
        scores = preds["scores"]

        keep = nms(boxes, scores, iou_threshold=0.4)
        boxes = boxes[keep]
        scores = scores[keep]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor="lime", linewidth=2))
            ax.text(x1, y1 - 5, f"{scores[i]:.2f}",
                    color="lime", fontsize=10, weight="bold")
        ax.axis("off")
        st.pyplot(fig)

# ======================================================
# 📈 TAB 2 — ANOMALY FORECASTING
# ======================================================
with tabs[1]:
    st.header("Forecast Next-Day Anomalies using ConvLSTM")

    st.markdown("Upload three CSV files representing the same solar array grid at three different inspection times:")

    t0_file = st.file_uploader("🟦 Upload Day 1 Grid (t₀)", type=["csv"], key="t0")
    t1_file = st.file_uploader("🟩 Upload Day 2 Grid (t₁)", type=["csv"], key="t1")
    t2_file = st.file_uploader("🟥 Upload Day 3 Grid (t₂ ground truth)", type=["csv"], key="t2")

    if t0_file and t1_file and t2_file:
        # Load model
        model = load_convlstm()

        # Load grids and masks
        grid0, mask0 = load_grid(t0_file)   # t₀
        grid1, mask1 = load_grid(t1_file)   # t₁
        grid2, mask2 = load_grid(t2_file)   # t₂ (ground truth)

        # Build input sequence exactly as in training: [t₀, t₁] -> predict t₂
        seq = torch.stack([grid0, grid1])   # shape (T=2, 1, H, W)

        with torch.no_grad():
            logits = model(seq)             # shape (1, 1, H, W)
            prob = torch.sigmoid(logits).squeeze().numpy()   # (H, W)

        # Threshold slider
        threshold = st.slider(
            "Classification threshold for anomaly (on predicted probability)",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.01
        )
        pred = (prob >= threshold).astype(int)

        # Use mask2 to ignore invalid cells (if you use -1 for no panel)
        mask = mask2.squeeze().numpy()
        true = grid2.squeeze().numpy()

        # ==========================
        # PROBABILITY HEATMAP
        # ==========================
        st.markdown("### Predicted Anomaly Probabilities for t₂")

        st.write(f"Probability range: {prob.min():.4f} → {prob.max():.4f}")
        st.write(f"Predicted anomaly count at threshold {threshold:.2f}: {pred[mask == 1].sum()} (valid cells only)")

        # Probability heatmap
        fig_prob, ax_prob = plt.subplots(figsize=(8, 8))
        im_prob = ax_prob.imshow(prob, cmap="hot", vmin=0.0, vmax=1.0)
        ax_prob.set_title("Predicted Anomaly Probability Heatmap (t₂)")
        ax_prob.axis("off")
        plt.colorbar(im_prob, ax=ax_prob)
        st.pyplot(fig_prob)

        # Binary map
        fig_bin, ax_bin = plt.subplots(figsize=(8, 8))
        ax_bin.imshow(pred, cmap="gray_r", vmin=0, vmax=1)
        ax_bin.set_title(f"Binary Prediction Map (threshold = {threshold:.2f})")
        ax_bin.axis("off")
        st.pyplot(fig_bin)

        # ==========================
        # CONFUSION MAP: pred t₂ vs true t₂
        # ==========================
        st.markdown("### Confusion Map (Predicted t₂ vs Ground Truth t₂)")

        true_arr = true.copy()
        pred_arr = pred.copy()

        # Only consider valid cells (mask == 1)
        valid = (mask == 1)
        # For visualization, we still want full grid, but confusion categories only mean something where mask==1

        tp = (true_arr == 1) & (pred_arr == 1) & valid
        fp = (true_arr == 0) & (pred_arr == 1) & valid
        fn = (true_arr == 1) & (pred_arr == 0) & valid
        tn = (true_arr == 0) & (pred_arr == 0) & valid

        conf_map = np.zeros_like(true_arr, dtype=int)
        conf_map[tn] = 0   # True Negative
        conf_map[fn] = 1   # False Negative
        conf_map[fp] = 2   # False Positive
        conf_map[tp] = 3   # True Positive

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([
            "#cccccc",  # 0 TN - grey
            "#ff4d4d",  # 1 FN - red
            "#ffd24d",  # 2 FP - yellow
            "#4dff4d"   # 3 TP - green
        ])

        fig_conf, ax_conf = plt.subplots(figsize=(8, 8))
        im_conf = ax_conf.imshow(conf_map, cmap=cmap, vmin=0, vmax=3)
        ax_conf.set_title("Confusion Map: Predicted t₂ vs Ground Truth t₂")
        ax_conf.axis("off")
        st.pyplot(fig_conf)

        # ==========================
        # METRICS (valid cells only)
        # ==========================
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        true_flat = true_arr[valid].flatten()
        pred_flat = pred_arr[valid].flatten()

        acc = accuracy_score(true_flat, pred_flat)
        prec = precision_score(true_flat, pred_flat, zero_division=0)
        rec = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)

        st.markdown("### Forecasting Performance (t₀, t₁ → predict t₂)")
        st.write(f"Accuracy:  {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall:    {rec:.4f}")
        st.write(f"F1 Score:  {f1:.4f}")

        st.markdown("""
        **Confusion Map Legend:**  
        - **Grey** = True Negative (correct normal panel)  
        - **Green** = True Positive (correct anomaly)  
        - **Yellow** = False Positive (false alarm)  
        - **Red** = False Negative (missed anomaly)  
        """)

        st.success("Prediction complete!")

