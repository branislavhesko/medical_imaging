import threading
import time
import io

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from omegaconf import OmegaConf
import torch

from medical_imaging.train_classifier import build_test_transforms

app = FastAPI()


_bedsores_classifier_thread = None

def get_bedsores_classifier_thread():
    global _bedsores_classifier_thread
    if _bedsores_classifier_thread is None or not _bedsores_classifier_thread.is_alive():
        print("Starting bedsores classifier thread")
        _bedsores_classifier_thread = BedSoresClassifierThread()
        _bedsores_classifier_thread.start()
    return _bedsores_classifier_thread


class BedSoresClassifierThread(threading.Thread):
    MODEL_PATH = "models/model_best.pt2"
    CONFIG_PATH = "models/model_best.yaml"
    UNUSED_TIMEOUT = 600 # 10 minutes
    
    def __init__(self):
        super().__init__()
        self.config = OmegaConf.load(self.CONFIG_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.transforms = build_test_transforms()
        self.model = None
        self.lock = threading.Lock()
        self.last_used = time.time()
        self.is_running = True
        self.stop_event = threading.Event()
        
    def run(self):
        while not self.stop_event.wait(timeout=10):
            with self.lock:
                if self.is_running and time.time() - self.last_used > self.UNUSED_TIMEOUT:
                    self.model = None
                    self.transforms = None
                    self.lock = None
                    self.last_used = None
                    self.is_running = False
                    self.stop_event.set()
                    print("Bedsores classifier thread stopped")
            
    @torch.no_grad()
    def predict(self, image: UploadFile = File(...)):
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self.model = self._load_model()
            image = Image.open(io.BytesIO(image)).convert("RGB")
            image = self.transforms(image)
            image = image.unsqueeze(0).to(self.config.device)
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1).squeeze(0)
            predicted_index = int(probabilities.argmax().item())
            classes = list(self.config.classes)
            return {
                "class": classes[predicted_index],
                "confidence": float(probabilities[predicted_index].item()),
                "probabilities": {
                    class_name: float(probability.item())
                    for class_name, probability in zip(classes, probabilities)
                },
            }
                
    def _load_model(self):
        self.model = torch.export.load(self.MODEL_PATH).module()
        self.model.to(self.config.device)
        return self.model


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