import time
from pathlib import Path

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interfaces.results import QuantizationResult
from src.quantization.base import BaseQuantizer

CALIBRATION_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "calibration_wikitext2_256.jsonl"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048


class AWQQuantizer(BaseQuantizer):
    """Quantizes a model with AWQ (4-bit weights, asymmetric, group_size=128) via llm-compressor.

    Static quantization: uses calibration data to identify and protect salient
    (activation-outlier) weight channels before rounding.
    """

    def quantize(self, model_path: str, output_dir: str) -> QuantizationResult:
        start = time.time()
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            ds = load_dataset("json", data_files=str(CALIBRATION_PATH), split="train")

            def tokenize(sample):
                return tokenizer(
                    sample["text"],
                    padding=False,
                    max_length=MAX_SEQUENCE_LENGTH,
                    truncation=True,
                    add_special_tokens=False,
                )

            ds = ds.map(tokenize)

            recipe = [
                AWQModifier(duo_scaling="both"),
                QuantizationModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                ),
            ]

            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=MAX_SEQUENCE_LENGTH,
                num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            )

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir, save_compressed=True)
            tokenizer.save_pretrained(output_dir)

            size_gb = sum(
                f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
            ) / (1024**3)

            return QuantizationResult(
                model=model_path,
                method="awq",
                output_path=output_dir,
                size_gb=size_gb,
                wall_clock_seconds=time.time() - start,
                success=True,
            )
        except Exception as e:
            return QuantizationResult(
                model=model_path,
                method="awq",
                output_path=output_dir,
                size_gb=0.0,
                wall_clock_seconds=time.time() - start,
                success=False,
                error=str(e),
            )
        finally:
            torch.cuda.empty_cache()
