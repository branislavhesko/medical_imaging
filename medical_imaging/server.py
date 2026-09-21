import threading
import time
import io

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from PIL import Image
from omegaconf import OmegaConf
import torch

from medical_imaging.train_classifier import build_test_transforms
from medical_imaging.bedsores_classifier_thread import BedSoresClassifierThread
from medical_imaging.ocr_thread import OCRThread
from medical_imaging.page_spec import parse_page_spec, select_pages

app = FastAPI()


_bedsores_classifier_thread = None
_ocr_engine_thread = None

def get_bedsores_classifier_thread():
    global _bedsores_classifier_thread
    if _bedsores_classifier_thread is None or not _bedsores_classifier_thread.is_alive():
        print("Starting bedsores classifier thread")
        _bedsores_classifier_thread = BedSoresClassifierThread()
        _bedsores_classifier_thread.start()
    return _bedsores_classifier_thread


def get_ocr_engine_thread():
    global _ocr_engine_thread
    if _ocr_engine_thread is None or not _ocr_engine_thread.is_alive():
        print("Starting OCR engine thread")
        _ocr_engine_thread = OCRThread()
        _ocr_engine_thread.start()
    return _ocr_engine_thread


# ---------------------------------------------------------------------------
# Inference entry points. Plain functions so the UI can call them in-process;
# the HTTP endpoints below are thin wrappers around them.
# ---------------------------------------------------------------------------
def run_bedsores(image_bytes: bytes) -> dict:
    return get_bedsores_classifier_thread().predict(image_bytes)


def run_ocr(file_bytes: bytes, pages: str = "") -> dict:
    """OCR an image or PDF.

    ``pages`` selects PDF pages: ``"1,3-5"``, ``"7"``, or empty/``"all"`` for
    every page. Ignored for single images. Raises :class:`ValueError` on a bad
    page selection.
    """
    page_numbers = parse_page_spec(pages)

    ocr = get_ocr_engine_thread()
    file_type = ocr._get_file_type_from_bytes(file_bytes)
    if file_type == "pdf":
        images = ocr._parse_pdf_into_images(file_bytes, page_numbers)
    else:
        images = [Image.open(io.BytesIO(file_bytes))]
        page_numbers = None

    images, page_numbers = select_pages(images, page_numbers)
    result = ocr._do_predict(images)
    return {"page_numbers": page_numbers, **result}


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/bedsores")
def detect_bedsores(image: UploadFile = File(...)):
    prediction = run_bedsores(image.file.read())
    return {"message": "Bedsores detection result", **prediction}


@app.post("/ocr")
def perform_ocr(file: UploadFile = File(...), pages: str = Form("")):
    try:
        result = run_ocr(file.file.read(), pages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"message": "OCR result", **result}


if __name__ == "__main__":
    # Standalone API server (optional). The UI (``python -m medical_imaging.ui``)
    # already mounts this app under ``/api``; run this only if you want the
    # API without the UI, and point the UI at it with BEDSORES_SERVER_URL.
    import uvicorn

    uvicorn.run("medical_imaging.server:app", host="0.0.0.0", port=8000, reload=False)
