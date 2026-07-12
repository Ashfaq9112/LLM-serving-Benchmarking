import json
import time
from pathlib import Path

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

from src.interfaces.results import QuantizationResult
from src.quantization.base import BaseQuantizer

CALIBRATION_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "calibration_wikitext2_256.jsonl"


class GPTQQuantizer(BaseQuantizer):
    """Quantizes a model with GPTQ (4-bit, group_size=128) via GPTQModel.

    Static quantization: uses calibration data to compute per-layer scales.
    """

    def quantize(self, model_path: str, output_dir: str) -> QuantizationResult:
        start = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            calibration_dataset = []
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line)["text"]
                    calibration_dataset.append(dict(tokenizer(text)))

            quant_config = QuantizeConfig(bits=4, group_size=128)
            model = GPTQModel.load(model_path, quant_config)
            model.quantize(calibration_dataset)

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save(output_dir)
            tokenizer.save_pretrained(output_dir)

            size_gb = sum(
                f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
            ) / (1024**3)

            return QuantizationResult(
                model=model_path,
                method="gptq",
                output_path=output_dir,
                size_gb=size_gb,
                wall_clock_seconds=time.time() - start,
                success=True,
            )
        except Exception as e:
            return QuantizationResult(
                model=model_path,
                method="gptq",
                output_path=output_dir,
                size_gb=0.0,
                wall_clock_seconds=time.time() - start,
                success=False,
                error=str(e),
            )
        finally:
            torch.cuda.empty_cache()
