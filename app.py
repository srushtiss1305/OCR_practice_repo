from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import uuid
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename

import cv2
import fitz  # PyMuPDF

from paddle_ocr import *


# -------------------- APP INIT --------------------
app = Flask(__name__)
CORS(app)

# -------------------- PATHS --------------------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed_images", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_transaction_id():
    return f"txn_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def pdf_to_images(pdf_path):
    """Convert PDF pages to OpenCV BGR images"""
    images = []
    doc = fitz.open(pdf_path)

    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img_np = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        images.append(img_np)

    return images


# -------------------- ROUTES --------------------
@app.route("/upload", methods=["POST"])
def upload():
    txn_id = generate_transaction_id()

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{txn_id}_{filename}")
        file.save(input_path)

        extracted_texts = []

        # -------- PDF --------
        if input_path.lower().endswith(".pdf"):
            pages = pdf_to_images(input_path)
            for img in pages:
                text = run_paddleocr(img)
                extracted_texts.append(text)

        # -------- IMAGE --------
        else:
            img = cv2.imread(input_path)
            if img is None:
                return jsonify({"error": "Failed to read image"}), 400

            extracted_texts.append(run_paddleocr(img))

        final_text = "\n\n".join(extracted_texts)

        text_path = os.path.join(
            OUTPUT_FOLDER, f"{txn_id}_extracted.txt"
        )
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        return jsonify({
            "status": "success",
            "transaction_id": txn_id,
            "text_file": text_path,
            "extracted_text": final_text
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "transaction_id": txn_id,
            "message": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Flask PaddleOCR API running")
    app.run(host="0.0.0.0", port=9000, threaded=False)