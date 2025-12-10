from PIL import ImageDraw

def draw_boxes(img, boxes, scores=None):
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

        if scores is not None:
            draw.text((x1, y1 - 10), f"{scores[i]:.2f}", fill="lime")

    return img 