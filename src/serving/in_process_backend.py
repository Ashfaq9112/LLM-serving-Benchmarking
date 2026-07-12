import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.interfaces.results import GenerationResult
from src.serving.base import BaseServingBackend


class InProcessBackend(BaseServingBackend):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", torch_dtype="auto"
        )

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        text = ""
        token_timestamps = []
        for chunk in streamer:
            token_timestamps.append(time.perf_counter())
            text += chunk
        thread.join()

        return GenerationResult(text=text, token_timestamps=token_timestamps)

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
