import json
import subprocess
import sys
import time

import requests

from src.interfaces.results import GenerationResult
from src.serving.base import BaseServingBackend


class VLLMServerBackend(BaseServingBackend):
    def __init__(self, port: int = 8000, startup_timeout_seconds: float = 300.0):
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds
        self.base_url = f"http://localhost:{port}"
        self.process: subprocess.Popen | None = None

    def load(self, model_path: str) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--port", str(self.port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = self.process.stdout.read()
                raise RuntimeError(
                    f"vLLM server process exited during startup (code {self.process.returncode}):\n{output}"
                )
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        self.unload()
        raise TimeoutError(f"vLLM server did not become healthy within {self.startup_timeout_seconds}s")

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self._served_model_name(),
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        text = ""
        token_timestamps = []
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            token_timestamps.append(time.perf_counter())
            text += chunk["choices"][0]["text"]

        return GenerationResult(text=text, token_timestamps=token_timestamps)

    def unload(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def _served_model_name(self) -> str:
        response = requests.get(f"{self.base_url}/v1/models", timeout=5)
        response.raise_for_status()
        return response.json()["data"][0]["id"]
