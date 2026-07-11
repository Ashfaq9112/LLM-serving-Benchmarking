
# Project Spec: Single Consumer-GPU LLM Serving Limits Benchmark

## Objective

Determine the actual serving ceiling of a single RTX 3090 (24GB) across a model-size ladder, and identify the point at which 4-bit quantization transitions from "throughput/power optimization" to "hard requirement for feasibility." Compare serving frameworks and quantization methods at each tier using self-quantized models (not pre-quantized community checkpoints), with real power measurement and task-grounded quality checks.

## Core Hypothesis (to prove or disprove with data)

*At what model size does 4-bit quantization stop being optional and become mandatory for a single 24GB GPU — and does the best-performing serving framework change once VRAM, not compute, becomes the binding constraint?*

## Hardware

- GPU: RTX 3090, 24GB VRAM, CUDA 13.2, Driver 595.71.05
- Log system RAM available too (needed for quantization steps, see below)

## Model Ladder

All from the **Qwen2.5-Instruct** family (consistent tokenizer/architecture = clean comparisons):

| Tier | Model | FP16 size | Expected outcome on 24GB |
|---|---|---|---|
| 1 — Sanity | Qwen2.5-1.5B-Instruct | ~3GB | Trivial at all precisions; use to debug harness |
| 2 — Comfortable | Qwen2.5-7B-Instruct | ~16GB | Fits FP16 with room for KV cache/batching |
| 3 — Transition | Qwen2.5-14B-Instruct | ~28GB | FP16 OOMs; INT8/4-bit fit easily |
| 4 — Ceiling | Qwen2.5-32B-Instruct | ~64GB | FP16 & INT8 both OOM; only 4-bit variants survive |

## Quantization Methodology — self-quantize, with one documented exception

**Self-quantize for Tiers 1–3 (1.5B, 7B, 14B):**
- **FP16** — baseline, no quantization
- **INT8** — bitsandbytes LLM.int8() at load time (dynamic, no separate calibration step, but log it as its own condition)
- **GPTQ 4-bit** — self-quantize via `AutoGPTQ` or `optimum-gptq`. Calibration set: 128 samples from WikiText-2, sequence length 2048 (standard practice — keep this exact calibration set constant across every model/tier for fair comparison)
- **AWQ 4-bit** — self-quantize via `AutoAWQ`, same calibration set as GPTQ
- **GGUF Q4_K_M** — convert HF checkpoint → FP16 GGUF via `llama.cpp/convert_hf_to_gguf.py`, then quantize via `llama-quantize` binary. This step is CPU/RAM-bound, not VRAM-bound, so it's actually your safest path for the 32B tier too.

**Tier 4 (32B) — the documented exception:**
Loading a 32B model at FP16 (~64GB) to *perform* GPTQ/AWQ calibration won't fit in 24GB VRAM even though the *quantized output* would. Options, in order of preference:
1. Try `device_map="auto"` with CPU offload during calibration (needs ~64GB+ system RAM, will be slow — check your system RAM first before attempting)
2. If that's infeasible: **self-quantize the GGUF path yourself** (CPU-bound, so no VRAM blocker) — this keeps at least one "I did this myself" quantization at this tier
3. For GPTQ/AWQ specifically at 32B, if step 1 fails: **use a pre-quantized community checkpoint** (e.g., from Qwen's official repo or a reputable uploader), clearly labeled in your results as "pre-quantized (not self-produced) — hardware constraint" rather than silently presenting it as your own work. This is a legitimate, well-reasoned engineering call, not a shortcut you need to hide — document *why* in your README (this is a good interview talking point on its own: recognizing and working around a real hardware constraint).

## Serving Frameworks

- HuggingFace `transformers` (baseline, unoptimized)
- vLLM
- llama.cpp server (for GGUF variants)
- TensorRT-LLM (Ampere/3090 has solid community support — actually attempt this rather than skip it)

## Calibration vs. Benchmark vs. Eval datasets (keep these three separate and don't mix them)

- **Calibration set** (for GPTQ/AWQ only): 128 samples, WikiText-2, seq len 2048 — fixed across all models
- **Benchmark prompt set** (for throughput/latency/power runs): 50 fixed prompts sampled from Alpaca or Dolly — same prompts used across every model/quant/framework combo
- **Quality eval set**: 100–200 examples for a structured JSON extraction task (e.g., extract name/date/amount from short invoice-like text snippets) — programmatically gradable (valid JSON? correct fields?)

## Metrics

**Per quantization run (Phase 1 — VRAM ceiling mapping):**
- Max context length and max batch size before OOM, per model × precision
- Explicit OOM log entries (don't skip failures — they're data)

**Per serving run (Phase 2 — throughput/latency/power):**
- Time-to-first-token (TTFT)
- Inter-token latency (ITL)
- Throughput (tokens/sec)
- Peak VRAM used
- Power draw, sampled every 1s via `nvidia-smi --query-gpu=power.draw --format=csv -l 1`, averaged over run
- **Derived: joules per 1,000 tokens** (headline efficiency metric)
- Concurrency levels: 1, 4, 8, 16

**Per quality run (Phase 3):**
- Valid-JSON-output rate
- Field-level accuracy rate
- Broken down by model × quantization level

## Repo / Notebook Structure

```
llm-serving-benchmark/
├── README.md
├── config/
│   └── models.yaml          # model names, tier, HF repo IDs
├── src/
│   ├── quantize_gptq.py
│   ├── quantize_awq.py
│   ├── quantize_gguf.py
│   ├── vram_ceiling.py
│   ├── power_logger.py       # nvidia-smi polling wrapper
│   ├── bench_runner.py       # runs throughput/latency across frameworks
│   ├── quality_eval.py       # JSON extraction task grading
│   └── aggregate_results.py
├── notebooks/
│   ├── 01_quantize_models.ipynb      # runs quantize_gptq/awq/gguf per tier, logs sizes/time
│   ├── 02_vram_ceiling.ipynb         # Phase 1: push context/batch until OOM, log breakpoints
│   ├── 03_benchmark_throughput_power.ipynb   # Phase 2: main benchmark sweep
│   ├── 04_quality_eval.ipynb         # Phase 3: JSON extraction accuracy per quant level
│   └── 05_aggregate_and_plot.ipynb   # Phase 4: charts + hypothesis conclusion
├── results/
│   ├── raw/                 # per-run CSV/JSON logs
│   └── figures/
└── data/
    ├── calibration_wikitext2_128.jsonl
    ├── benchmark_prompts_50.jsonl
    └── json_extraction_eval_150.jsonl
```

Each notebook imports from `src/` rather than containing raw logic inline — keeps the notebooks readable as a walkthrough while the actual reusable code lives in testable modules.

## Phased Build Order

1. **Setup**: environment, `models.yaml`, fixed datasets committed to repo first (before any quantization) so every run is reproducible against the same inputs
2. **Notebook 01**: quantize all feasible model×method combos, log wall-clock time and output size for each (this alone is useful data — quantization cost is rarely reported)
3. **Notebook 02**: VRAM ceiling sweep per surviving combo
4. **Notebook 03**: throughput/latency/power sweep across frameworks, at concurrency 1/4/8/16, only on combos that survived Notebook 02
5. **Notebook 04**: quality eval across quant levels
6. **Notebook 05**: aggregate everything, generate the throughput-per-watt chart, VRAM ceiling table, quality-degradation chart, and write the final hypothesis conclusion with actual numbers

## Deliverables

- Public GitHub repo with the structure above, reproducible end-to-end
- README with: objective, hypothesis, methodology, headline charts, and a clearly stated conclusion (supported or refuted, with numbers)
- Explicit note in README about the Tier 4 quantization exception and why it was made