import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.interfaces.results import QuantizationResult
from src.quantization.base import BaseQuantizer

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"
# Path(__file__) is this file's own path (src/quantization/gguf_quantizer.py). .parent.parent.parent walks up three levels: quantization/ → src/ → repo root, then down into config/models.yaml. This mirrors the CALIBRATION_PATH pattern in gptq_quantizer.py/awq_quantizer.py — locate a project file relative to this script's location so it works regardless of what directory you run Python from.
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _LLAMACPP_CONFIG = yaml.safe_load(f)["llamacpp"]


class GGUFQuantizer(BaseQuantizer):
    """Quantizes a model to GGUF Q4_K_M by shelling out to llama.cpp's own tools.

    Two subprocess steps, CPU-only, no GPU involved:
      1. convert_hf_to_gguf.py: HF checkpoint -> GGUF f16
      2. llama-quantize: GGUF f16 -> GGUF Q4_K_M
    """

    def quantize(self, model_path: str, output_dir: str) -> QuantizationResult:
        start = time.time()
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            f16_path = Path(output_dir) / "model-f16.gguf"
            q4_path = Path(output_dir) / "model-Q4_K_M.gguf"

            subprocess.run(
                [
                    sys.executable,
                    _LLAMACPP_CONFIG["convert_script"],
                    model_path,
                    "--outfile", str(f16_path),
                    "--outtype", "f16",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            subprocess.run(
                [
                    _LLAMACPP_CONFIG["quantize_binary"],
                    str(f16_path),
                    str(q4_path),
                    "Q4_K_M",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            f16_path.unlink()

            size_gb = q4_path.stat().st_size / (1024**3)

            return QuantizationResult(
                model=model_path,
                method="gguf_q4_k_m",
                output_path=str(q4_path),
                size_gb=size_gb,
                wall_clock_seconds=time.time() - start,
                success=True,
            )
        except subprocess.CalledProcessError as e:
            return QuantizationResult(
                model=model_path,
                method="gguf_q4_k_m",
                output_path=output_dir,
                size_gb=0.0,
                wall_clock_seconds=time.time() - start,
                success=False,
                error=e.stderr,
            )
        except Exception as e:
            return QuantizationResult(
                model=model_path,
                method="gguf_q4_k_m",
                output_path=output_dir,
                size_gb=0.0,
                wall_clock_seconds=time.time() - start,
                success=False,
                error=str(e),
            )
