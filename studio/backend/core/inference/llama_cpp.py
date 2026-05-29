import logging
import os
import subprocess
import threading
from typing import Optional

from llama_cpp import LlamaCpp

logger = logging.getLogger(__name__)

class LlamaCppModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.llama_cpp = LlamaCpp(model_path, device)

    def load_model(self):
        try:
            self.llama_cpp.load_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError("llama-server failed to start. Check that the GGUF file is valid and you have enough memory.")

    def unload_model(self):
        self.llama_cpp.unload_model()

    def get_model_info(self):
        return self.llama_cpp.get_model_info()

    def get_device_info(self):
        return self.llama_cpp.get_device_info()

    def get_system_info(self):
        return self.llama_cpp.get_system_info()

    def get_model_architecture(self):
        return self.llama_cpp.get_model_architecture()

def load_model(model_path: str, device: str = "cpu") -> Optional[LlamaCppModel]:
    try:
        model = LlamaCppModel(model_path, device)
        model.load_model()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None