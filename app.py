from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import uuid
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename

from preprocess import pdf_to_images, deskew_image, preprocess_image
from paddleocr import run_ocr
from postprocess import clean_text

# -------------------- APP INIT --------------------
app = Flask(__name__)
CORS(app)

# -------------------- PATHS --------------------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed_images")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_transaction_id():
    return f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

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
        input_path = os.path.join(
            UPLOAD_FOLDER, f"{txn_id}_{filename}"
        )
        file.save(input_path)

        extracted_texts = []

        if input_path.lower().endswith(".pdf"):
            pages = pdf_to_images(input_path, txn_id)
            for page_path in pages:
                deskewed = deskew_image(page_path, txn_id)
                pre = preprocess_image(deskewed, txn_id)
                extracted_texts.append(run_ocr(pre))
        else:
            deskewed = deskew_image(input_path, txn_id)
            pre = preprocess_image(deskewed, txn_id)
            extracted_texts.append(run_ocr(pre))

        final_text = clean_text("\n\n".join(extracted_texts))

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