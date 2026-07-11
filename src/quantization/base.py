from abc import ABC, abstractmethod

from src.interfaces.results import QuantizationResult


class BaseQuantizer(ABC):
    @abstractmethod
    def quantize(self, model_path: str, output_dir: str) -> QuantizationResult:
        ...
