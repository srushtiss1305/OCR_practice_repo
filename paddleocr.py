import cv2
import numpy as np
from typing import List, Dict
from paddleocr import PaddleOCR
from deskew import determine_skew


_paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec_batch_num=16
)

# -----------rotate image with no crop---------------
def rotate_image(image, angle):
    if angle == 0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

# ----------preprocessing------------
def preprocess_for_paddleocr(img_bgr):


    # --- DESKEW ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    skew_angle = determine_skew(gray)
    img = rotate_image(img_bgr, skew_angle)

    # --- Resize ---
    h, w = img.shape[:2]
    if h < 700:
        scale = 700 / h
        img = cv2.resize(img, (int(w * scale), 700), interpolation=cv2.INTER_CUBIC)
    elif h > 1200:
        scale = 1200 / h
        img = cv2.resize(img, (int(w * scale), 1200), interpolation=cv2.INTER_AREA)

    return img

#-------------------normalization---------------
def normalize_paddle_result(ocr_result):
    texts = ocr_result["rec_texts"]
    scores = ocr_result["rec_scores"]
    polys = ocr_result["rec_polys"]

    boxes = []

    for text, conf, poly in zip(texts, scores, polys):
        if not text.strip():
            continue

        poly = np.array(poly)
        xs, ys = poly[:, 0], poly[:, 1]

        boxes.append({
            "text": text.strip(),
            "conf": float(conf),
            "x_min": float(xs.min()),
            "y_min": float(ys.min()),
            "x_max": float(xs.max()),
            "y_max": float(ys.max()),
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "height": float(ys.max() - ys.min())
        })

    return boxes

#-----------group boxes into lines--------------
def group_boxes_into_lines(
    boxes: List[Dict],
    y_threshold_ratio: float = 0.6
) -> List[str]:

    boxes = sorted(boxes, key=lambda b: b["cy"])

    lines = []
    current_line = []
    current_y = None

    for box in boxes:
        if current_y is None:
            current_line = [box]
            current_y = box["cy"]
            continue

        y_thresh = box["height"] * y_threshold_ratio

        if abs(box["cy"] - current_y) <= y_thresh:
            current_line.append(box)
            current_y = (current_y + box["cy"]) / 2
        else:
            current_line = sorted(current_line, key=lambda b: b["x_min"])
            lines.append(" ".join(b["text"] for b in current_line))
            current_line = [box]
            current_y = box["cy"]

    if current_line:
        current_line = sorted(current_line, key=lambda b: b["x_min"])
        lines.append(" ".join(b["text"] for b in current_line))

    return lines

# -----------main-----------
def run_paddleocr(img_bgr) -> str:

    pre_img = preprocess_for_paddleocr(img_bgr)

    if hasattr(_paddle_ocr, "predict"):
        ocr_raw = _paddle_ocr.predict(pre_img)
    else:
        ocr_raw = _paddle_ocr.ocr(pre_img)

    boxes = normalize_paddle_result(ocr_raw[0])
    lines = group_boxes_into_lines(boxes)

    return "\n".join(lines)
