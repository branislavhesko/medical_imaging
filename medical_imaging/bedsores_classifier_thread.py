import concurrent.futures
import queue
import threading
import time
import io

from fastapi import UploadFile, File
from PIL import Image
from omegaconf import OmegaConf
import torch

from medical_imaging.train_classifier import build_test_transforms


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
        self.tasks = queue.Queue()
        
    def run(self):
        while not self.stop_event.is_set():
            try:
                task, task_data, future = self.tasks.get(timeout=10)
            
            except queue.Empty:
                if self.last_used < time.time() - self.UNUSED_TIMEOUT:
                    self.model = None
                    self.stop_event.set()
                    break
            try:
                if task == "predict":
                    result = self._do_predict(task_data)
                    future.set_result(result)
                else:
                    future.set_exception(ValueError(f"Unknown task: {task}"))
            except Exception as e:
                future.set_exception(e)
                print(f"Error predicting bedsores: {e}")
            finally:
                self.tasks.task_done()

    def predict(self, image: UploadFile):
        future = concurrent.futures.Future()
        self.tasks.put(("predict", image, future))
        return future.result()
                
    @torch.no_grad()
    def _do_predict(self, image: UploadFile):
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