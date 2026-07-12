from abc import ABC, abstractmethod

from src.interfaces.results import GenerationResult


class BaseServingBackend(ABC):
    @abstractmethod
    def load(self, model_path: str) -> None:
        ...

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...
