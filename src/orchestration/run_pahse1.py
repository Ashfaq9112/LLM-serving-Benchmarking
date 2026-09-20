import dataclasses
import json
from pathlib import Path

from src.evaluation.evaluator import Evaluator
from src.quantization.awq_quantizer import AWQQuantizer
from src.quantization.gguf_quantizer import GGUFQuantizer
from src.quantization.gptq_quantizer import GPTQQuantizer
from src.serving.in_process_backend import InProcessBackend
from src.serving.vllm_backend import VLLMServerBackend
from src.serving.llamacpp_backend import LlamaCppServerBackend


MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR = Path("results/raw/models")
RESULTS_DIR = Path("results/raw")
RESULTS_PATH = RESULTS_DIR / "phase1_results.json"


def save_result(method: str, framework: str, vram_result, benchmark_result) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    data[f"{method}_{framework}"] = {
        "vram_ceiling": dataclasses.asdict(vram_result),
        "benchmark": [dataclasses.asdict(r) for r in benchmark_result],
    }
    RESULTS_PATH.write_text(json.dumps(data, indent=2))


def run_orchestrator():
    awq_path = str(CHECKPOINT_DIR / "awq")
    gptq_path = str(CHECKPOINT_DIR / "gptq")

    # fp16: no quantization step, the Hub checkpoint already is fp16
    result = AWQQuantizer().quantize(MODEL, awq_path)
    print("awq quantize:", result)

    result = GPTQQuantizer().quantize(MODEL, gptq_path)
    print("gptq quantize:", result)

    gguf_result = GGUFQuantizer().quantize(MODEL, str(CHECKPOINT_DIR / "gguf"))
    print("gguf quantize:", gguf_result)
    gguf_path = gguf_result.output_path

    gguf_f16_result = GGUFQuantizer().quantize_f16(MODEL, str(CHECKPOINT_DIR / "gguf_f16"))
    print("gguf f16 quantize:", gguf_f16_result)
    gguf_f16_path = gguf_f16_result.output_path
    return  # quantize-only for now; evaluation loops below are not run yet

    for model_path, method in [(MODEL, "fp16"), (awq_path, "awq"), (gptq_path, "gptq")]:
        backend = InProcessBackend()
        backend.load(model_path)
        evaluator = Evaluator(backend, model_path, method, "inprocess")
        print(f"[{method}] running vram ceiling")
        vram_result = evaluator.run_vram_ceiling()
        print(f"[{method}] running benchmark")
        benchmark_result = evaluator.run_benchmark()
        save_result(method, "inprocess", vram_result, benchmark_result)
        backend.unload()

    for model_path, method in [(MODEL, "fp16"), (awq_path, "awq"), (gptq_path, "gptq")]:
        backend = VLLMServerBackend()
        backend.load(model_path)
        evaluator = Evaluator(backend, model_path, method, "vllm")
        print(f"[{method}] vllm running vram ceiling")
        vram_result = evaluator.run_vram_ceiling()
        print(f"[{method}] vllm running benchmark")
        benchmark_result = evaluator.run_benchmark()
        save_result(method, "vllm", vram_result, benchmark_result)
        backend.unload()

    for model_path, method in [(gguf_f16_path, "fp16"), (gguf_path, "gguf_q4_k_m")]:
        backend = LlamaCppServerBackend()
        backend.load(model_path)
        evaluator = Evaluator(backend, model_path, method, "llamacpp")
        print(f"[{method}] llamacpp running vram ceiling")
        vram_result = evaluator.run_vram_ceiling()
        print(f"[{method}] llamacpp running benchmark")
        benchmark_result = evaluator.run_benchmark()
        save_result(method, "llamacpp", vram_result, benchmark_result)
        backend.unload()

if __name__ == "__main__":
    run_orchestrator()







