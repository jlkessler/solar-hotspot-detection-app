from huggingface_hub import hf_hub_download
import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


REPO_ID = "jlkessler/hotspot_detector"
FILENAME = "hotspot_detector.pth"


@st.cache_resource
def load_hotspot_model():
    # HuggingFace will download the file if missing
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        cache_dir="."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2

    # Load Faster R-CNN base
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🔆 Solar Panel Hotspot Detection")
st.write("Upload a thermal image of a solar array to detect hotspot anomalies using a Faster R-CNN model.")

uploaded = st.file_uploader("Upload a thermal image (JPG/PNG)", type=["jpg", "jpeg", "png"])

model, device = load_hotspot_model()

if uploaded is not None:
    # Hardcoded thresholds
    score_threshold = 0.50     # confidence threshold
    iou_threshold = 0.40       # NMS IoU threshold

    img = Image.open(uploaded).convert("RGB")

    st.subheader("Original Image")
    st.image(img)

    transform = T.Compose([T.ToTensor()])
    x = transform(img).to(device)

    with torch.no_grad():
        outputs = model([x])[0]

    boxes = outputs["boxes"]
    scores = outputs["scores"]

    # Filter by score
    keep_score = scores >= score_threshold
    boxes = boxes[keep_score]
    scores = scores[keep_score]

    if len(boxes) == 0:
        st.warning(f"No hotspots detected above score threshold ({score_threshold}).")
    else:
        # Apply NMS
        keep_idx = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].cpu()
            conf = scores[i].item()

            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="lime",
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"{conf:.2f}",
                color="lime",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
            )

        ax.set_axis_off()
        st.subheader("Detected Hotspots")
        st.pyplot(fig)

        st.write(f"Detected **{len(boxes)}** hotspot(s).")
