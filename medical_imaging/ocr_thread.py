import concurrent.futures
import io
import queue
import threading
import time

from fastapi import UploadFile, File
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


class OCRThread(threading.Thread):
    MODEL_ID = 'jinaai/jina-ocr-v1'
    UNUSED_TIMEOUT = 600 # 10 minutes

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = None
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
                continue

            try:
                if task == "predict":
                    result = self._do_predict(task_data)
                    future.set_result(result)
                else:
                    future.set_exception(ValueError(f"Unknown task: {task}"))
            except Exception as e:
                future.set_exception(e)
                print(f"Error performing OCR: {e}")
            finally:
                self.tasks.task_done()

    def predict(self, image: bytes | UploadFile):
        if hasattr(image, "file"):
            image = image.file.read()
        future = concurrent.futures.Future()
        self.tasks.put(("predict", image, future))
        return future.result()

    @torch.no_grad()
    def _do_predict(self, image: bytes):
        with self.lock:
            self.last_used = time.time()
            if self.model is None or self.processor is None:
                self._load_model()
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            
            inputs = self.processor.prepare_ocr_inputs(image, device=self.device)
            output = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            text = self.processor.decode_ocr(output, inputs['input_ids'])
            return {"text": text}

    def _load_model(self):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
            ).to(self.device)
        return self.model

    def _initialize(self):
        return self._load_model()

    def _parse_pdf_into_images(self, pdf: UploadFile):
        pass