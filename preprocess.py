import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

# -------------------- PATHS --------------------
BASE_DIR = os.getcwd()
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed_images")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------- HELPERS --------------------
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def pdf_to_images(pdf_path, txn_id):
    images = []
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        out_path = os.path.join(
            OUTPUT_FOLDER, f"{txn_id}_page_{i}.jpg"
        )
        cv2.imwrite(out_path, np.array(img))
        images.append(out_path)

    doc.close()
    return images


def deskew_image(image_path, txn_id):
    img = safe_imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            deg = np.degrees(theta) - 90
            if -45 < deg < 45:
                angles.append(deg)
        if angles:
            angle = np.median(angles)

    if abs(angle) > 0.5:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    out = os.path.join(
        OUTPUT_FOLDER, f"{txn_id}_deskewed.jpg"
    )
    cv2.imwrite(out, img)
    return out


def preprocess_image(image_path, txn_id):
    img = safe_imread(image_path)

    h, w = img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out = os.path.join(
        OUTPUT_FOLDER, f"{txn_id}_preprocessed.jpg"
    )
    cv2.imwrite(out, gray)
    return out
