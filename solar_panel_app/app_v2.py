import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_hotspot_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2  # background + hotspot

    # Base Faster R-CNN with ResNet-50 FPN
    model = fasterrcnn_resnet50_fpn(weights=None)  # we will load your own weights

    # Replace the classifier head to match num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load your trained weights
    state_dict = torch.load("models/hotspot_detector.pth", map_location=device)
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
    # Hard-coded thresholds (edit as desired)
    score_threshold = 0.50     # Only keep detections above 50% confidence
    iou_threshold = 0.40       # NMS overlap threshold

    img = Image.open(uploaded).convert("RGB")

    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    transform = T.Compose([T.ToTensor()])
    x = transform(img).to(device)


    with torch.no_grad():
        outputs = model([x])[0]

    boxes = outputs["boxes"]
    scores = outputs["scores"]

    # Filter boxes by score
    keep_score = scores >= score_threshold
    boxes = boxes[keep_score]
    scores = scores[keep_score]

    if len(boxes) == 0:
        st.warning(f"No hotspots detected above score threshold ({score_threshold}).")
    else:
        # Apply NMS using the hardcoded IoU threshold
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

