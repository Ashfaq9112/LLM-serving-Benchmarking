from dataclasses import dataclass


@dataclass
class QuantizationResult:
    model: str
    method: str  # "fp16" | "int8" | "gptq" | "awq" | "gguf_q4_k_m"
    output_path: str
    size_gb: float
    wall_clock_seconds: float
    success: bool
    error: str | None = None


@dataclass
class VRAMCeilingResult:
    model: str
    method: str
    max_context_len: int | None  # None if OOM at minimum
    max_batch_size: int | None
    oom: bool
    oom_detail: str | None = None


@dataclass
class GenerationResult:
    text: str
    token_timestamps: list[float]  # wall-clock time each output token was produced


@dataclass
class BenchmarkResult:
    model: str
    method: str
    framework: str
    concurrency: int
    ttft_ms: float
    itl_ms: float
    throughput_tok_s: float
    peak_vram_gb: float
    avg_power_watts: float
    joules_per_1k_tokens: float


@dataclass
class QualityResult:
    model: str
    method: str
    valid_json_rate: float
    field_accuracy_rate: float
