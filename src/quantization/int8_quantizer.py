import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.interfaces.results import QuantizationResult
from src.quantization.base import BaseQuantizer


class Int8Quantizer(BaseQuantizer):
    """Loads a model with bitsandbytes LLM.int8() and saves it.

    Dynamic quantization: no calibration data, scale factors computed at load time.
    """

    def quantize(self, model_path: str, output_dir: str) -> QuantizationResult:
        start = time.time()
        try:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            size_gb = sum(
                f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
            ) / (1024**3)

            return QuantizationResult(
                model=model_path,
                method="int8",
                output_path=output_dir,
                size_gb=size_gb,
                wall_clock_seconds=time.time() - start,
                success=True,
            )
        except Exception as e:
            return QuantizationResult(
                model=model_path,
                method="int8",
                output_path=output_dir,
                size_gb=0.0,
                wall_clock_seconds=time.time() - start,
                success=False,
                error=str(e),
            )
        finally:
            torch.cuda.empty_cache()
