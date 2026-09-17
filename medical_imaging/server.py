import threading
import time
import io

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from omegaconf import OmegaConf
import torch

from medical_imaging.train_classifier import build_test_transforms
from medical_imaging.bedsores_classifier_thread import BedSoresClassifierThread
from medical_imaging.ocr_thread import OCRThread

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


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/bedsores")
def detect_bedsores(image: UploadFile = File(...)):
    prediction = get_bedsores_classifier_thread().predict(image.file.read())
    return {"message": "Bedsores detection result", **prediction}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("medical_imaging.server:app", host="0.0.0.0", port=8000, reload=False)