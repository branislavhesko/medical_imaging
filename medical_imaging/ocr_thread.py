import concurrent.futures
import gc
import io
import logging
import queue
import threading
import time

from fastapi import UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OCRThread(threading.Thread):
    MODEL_ID = 'jinaai/jina-ocr-v1'
    UNUSED_TIMEOUT = 600 # 10 minutes

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device} for model {self.MODEL_ID}")
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
                self._unload_if_idle()
                continue

            try:
                if task == "predict":
                    result = self._do_predict(task_data)
                    future.set_result(result)
                else:
                    future.set_exception(ValueError(f"Unknown task: {task}"))
            except Exception as e:
                future.set_exception(e)
                logger.exception("Error performing OCR")
            finally:
                self.tasks.task_done()

    def predict(self, image: bytes | UploadFile):
        if hasattr(image, "file"):
            image = image.file.read()
        future = concurrent.futures.Future()
        self.tasks.put(("predict", image, future))
        return future.result()

    def _unload_if_idle(self):
        """Drop the ~7 GB of weights once nobody has used them for a while.

        The thread itself keeps running so callers never see a dead thread;
        the next request simply reloads the model.
        """
        if self.model is None or time.time() - self.last_used < self.UNUSED_TIMEOUT:
            return
        with self.lock:
            # Re-check: a request may have arrived while we waited for the lock.
            if self.model is None or time.time() - self.last_used < self.UNUSED_TIMEOUT:
                return
            logger.info(f"Unloading {self.MODEL_ID} after {self.UNUSED_TIMEOUT}s idle")
            self.model = None
            self.processor = None
            gc.collect()

    @torch.no_grad()
    def _do_predict(self, uploaded_file: UploadFile | bytes | Image.Image | list[Image.Image]):
        with self.lock:
            try:
                return self._predict_locked(uploaded_file)
            finally:
                # Count the whole job as "use" so a long PDF isn't unloaded
                # the moment it finishes.
                self.last_used = time.time()

    def _predict_locked(self, uploaded_file):
        self.last_used = time.time()
        if self.model is None or self.processor is None:
            self._load_model()
        
        if isinstance(uploaded_file, list):
            images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in uploaded_file]
        elif isinstance(uploaded_file, Image.Image):
            images = [uploaded_file.convert("RGB")]
        else:
            if hasattr(uploaded_file, "file"):
                uploaded_file.file.seek(0)
                raw_bytes = uploaded_file.file.read()
                uploaded_file.file.seek(0)
            elif hasattr(uploaded_file, "read"):
                raw_bytes = uploaded_file.read()
            elif isinstance(uploaded_file, (bytes, bytearray)):
                raw_bytes = bytes(uploaded_file)
            else:
                raise TypeError(f"Unsupported input type: {type(uploaded_file)}")

            file_type = self._get_file_type_from_bytes(raw_bytes)
            if file_type == "pdf":
                images = self._parse_pdf_into_images(raw_bytes)
            else:
                images = [Image.open(io.BytesIO(raw_bytes)).convert("RGB")]
        
        if not images:
            return {"text": "", "pages": []}

        texts = []
        for img in tqdm(images):
            inputs = self.processor.prepare_ocr_inputs(img, device=self.device)
            output = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            text = self.processor.decode_ocr(output, inputs['input_ids'])
            texts.append(text)

        return {"text": "\n\n".join(texts), "pages": texts}

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

    def _get_file_type_from_bytes(self, data: bytes | bytearray | UploadFile) -> str:
        if hasattr(data, "file"):
            data.file.seek(0)
            header = data.file.read(1024)
            data.file.seek(0)
        elif hasattr(data, "read") and hasattr(data, "seek"):
            pos = data.tell() if hasattr(data, "tell") else 0
            header = data.read(1024)
            data.seek(pos)
        elif isinstance(data, (bytes, bytearray)):
            header = data[:1024]
        else:
            raise TypeError(f"Unsupported input type for file type detection: {type(data)}")

        if header.startswith(b"%PDF-"):
            return "pdf"
        return "image"

    def _parse_pdf_into_images(
        self, pdf: UploadFile | bytes, page_numbers: list[int] | None = None
    ) -> list[Image.Image]:
        """Rasterise a PDF. ``page_numbers`` (1-based) limits rendering to the
        span covering those pages; the returned list is still indexed by
        absolute page number (unrendered pages are ``None``)."""
        if hasattr(pdf, "file"):
            pdf.file.seek(0)
            pdf_bytes = pdf.file.read()
            pdf.file.seek(0)
        elif isinstance(pdf, (bytes, bytearray)):
            pdf_bytes = bytes(pdf)
        elif hasattr(pdf, "read"):
            pdf_bytes = pdf.read()
        else:
            raise TypeError(f"Unsupported pdf type: {type(pdf)}")
        if not page_numbers:
            return convert_from_bytes(pdf_bytes)
        first, last = min(page_numbers), max(page_numbers)
        rendered = convert_from_bytes(pdf_bytes, first_page=first, last_page=last)
        # Pad so that list index == page number - 1 for select_pages().
        return [None] * (first - 1) + rendered