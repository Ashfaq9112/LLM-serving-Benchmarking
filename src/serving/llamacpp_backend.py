import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import yaml

from src.interfaces.results import GenerationResult
from src.serving.base import BaseServingBackend

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _LLAMACPP_CONFIG = yaml.safe_load(f)["llamacpp"]


class LlamaCppServerBackend(BaseServingBackend):
    def __init__(self, port: int = 8001, startup_timeout_seconds: float = 300.0):
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds
        self.base_url = f"http://localhost:{port}"
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def load(self, model_path: str) -> None:
        self.log_file = tempfile.NamedTemporaryFile(
            mode="w+", prefix="llamacpp_server_", suffix=".log", delete=False
        )
        self.process = subprocess.Popen(
            [
                _LLAMACPP_CONFIG["server_binary"],
                "-m", model_path,
                "--port", str(self.port),
                "-ngl", "999",
            ],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_file.seek(0)
                output = self.log_file.read()
                raise RuntimeError(
                    f"llama.cpp server process exited during startup (code {self.process.returncode}):\n{output}"
                )
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        self.unload()
        raise TimeoutError(f"llama.cpp server did not become healthy within {self.startup_timeout_seconds}s")

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
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
        if self.log_file is not None:
            self.log_file.close()
            Path(self.log_file.name).unlink()
            self.log_file = None
