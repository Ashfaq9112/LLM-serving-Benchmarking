"""Builds the fixed calibration and benchmark datasets from HF hub, once,
so every later run reads the same frozen samples instead of live HF data."""
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_calibration_set(num_samples: int = 128, seq_len: int = 2048) -> None:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in ds if len(ex["text"].strip()) > 0]
    samples = texts[:num_samples]
    out_path = DATA_DIR / "calibration_wikitext2_128.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for text in samples:
            f.write(json.dumps({"text": text[:seq_len]}) + "\n")
    print(f"Wrote {len(samples)} calibration samples to {out_path}")


def build_benchmark_prompts(num_prompts: int = 50) -> None:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = [ex["instruction"] for ex in ds][:num_prompts]
    out_path = DATA_DIR / "benchmark_prompts_50.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}) + "\n")
    print(f"Wrote {len(prompts)} benchmark prompts to {out_path}")


if __name__ == "__main__":
    build_calibration_set()
    build_benchmark_prompts()
