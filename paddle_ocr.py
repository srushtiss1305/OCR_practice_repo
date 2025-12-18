import os
from paddleocr import PaddleOCR
from threading import Lock

# -------------------- ENV SAFETY --------------------
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["OMP_NUM_THREADS"] = "1"

# -------------------- OCR INIT --------------------
ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    text_det_limit_side_len=960
)

# PaddleOCR is NOT thread-safe
ocr_lock = Lock()


def run_ocr(image_path):
    with ocr_lock:
        result = ocr.ocr(image_path, cls=False)

    lines = []

    # result = [ [ [box, (text, score)], ... ] ]
    for page in result:
        for line in page:
            if len(line) >= 2:
                text = line[1][0]
                lines.append(text)

    return "\n".join(lines)
