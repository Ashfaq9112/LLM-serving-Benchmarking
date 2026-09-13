import concurrent.futures

import torch
from src.evaluation.power_logger import PowerLogger
from src.interfaces.results import VRAMCeilingResult
from src.serving.base import BaseServingBackend
from src.evaluation.vrampeaklogger import VRAMPeakLogger
from src.serving.in_process_backend import InProcessBackend
from src.interfaces.results import BenchmarkResult
import time

# Rough token-count proxy: repeating a short word gets close enough to the target
# token count for OOM-sweep purposes without needing tokenizer access through the
# BaseServingBackend interface (which only exposes generate(), not the tokenizer).
CONTEXT_LEN_CANDIDATES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
BATCH_SIZE_CANDIDATES = [1, 2, 4, 8, 16, 32]
BENCHMARK_CONCURRENCY_LEVELS = [1, 4, 8, 16]

class Evaluator:
    def __init__(self, backend: BaseServingBackend, model: str, method: str, framework: str):
        self.backend = backend
        self.model = model
        self.method = method
        self.framework = framework

    def run_vram_ceiling(self) -> VRAMCeilingResult:
        max_context_len, context_oom_detail = self._sweep_context_len()
        if context_oom_detail is not None:
            return VRAMCeilingResult(
                model=self.model,
                method=self.method,
                max_context_len=max_context_len,
                max_batch_size=None,
                oom=True,
                oom_detail=context_oom_detail,
            )

        max_batch_size, batch_oom_detail = self._sweep_batch_size()
        return VRAMCeilingResult(
            model=self.model,
            method=self.method,
            max_context_len=max_context_len,
            max_batch_size=max_batch_size,
            oom=batch_oom_detail is not None,
            oom_detail=batch_oom_detail,
        )

    def _sweep_context_len(self) -> tuple[int | None, str | None]:
        last_success = None
        for context_len in CONTEXT_LEN_CANDIDATES:
            prompt = "hello " * context_len
            try:
                self.backend.generate(prompt, max_new_tokens=16)
                last_success = context_len
            except Exception as e:
                if self._is_oom(e):
                    return last_success, str(e)
                raise
        return last_success, None

    def _sweep_batch_size(self) -> tuple[int | None, str | None]:
        prompt = "Explain the theory of relativity in a few sentences."
        last_success = None
        for batch_size in BATCH_SIZE_CANDIDATES:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
                    futures = [
                        pool.submit(self.backend.generate, prompt, 64)
                        for _ in range(batch_size)
                    ]
                    for future in futures:
                        future.result()
                last_success = batch_size
            except Exception as e:
                if self._is_oom(e):
                    return last_success, str(e)
                raise
        return last_success, None


    def _itl_for_one(self,timestamps: list[float]) -> float:
        gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        return sum(gaps) / len(gaps)
    
    def run_at_concurrency(self,concurrency):
        prompt = "Explain the theory of relativity in a few sentences."
        t0 = time.time()
        if isinstance(self.backend, InProcessBackend):
            torch.cuda.reset_peak_memory_stats()
        else:
            vram_logger = VRAMPeakLogger()
            vram_logger.start()
        power_logger = PowerLogger()
        power_logger.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self.backend.generate, prompt, 64) for _ in range(concurrency)]
            results = [f.result() for f in futures]  # list of GenerationResult

        total_duration = time.time() - t0
        if isinstance(self.backend, InProcessBackend):
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        else:
            peak_vram_gb = vram_logger.stop()
        avg_power_watts = power_logger.stop()
        
        
        ttft_ms = sum((r.token_timestamps[0] - t0) * 1000 for r in results) / len(results)
        itl_ms = sum(self._itl_for_one(r.token_timestamps) for r in results) / len(results) * 1000
        total_tokens = sum(len(r.token_timestamps) for r in results)
        throughput_tok_s = total_tokens / total_duration
        joules_per_1k_tokens = (avg_power_watts * total_duration) / (total_tokens / 1000)
        return BenchmarkResult(
            model=self.model,
            method=self.method,
            framework=self.framework,
            concurrency=concurrency,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            throughput_tok_s=throughput_tok_s,
            peak_vram_gb=peak_vram_gb,
            avg_power_watts=avg_power_watts,
            joules_per_1k_tokens=joules_per_1k_tokens,
        )


            
    def run_benchmark(self):
        return [self.run_at_concurrency(c) for c in BENCHMARK_CONCURRENCY_LEVELS]





    @staticmethod
    def _is_oom(e: Exception) -> bool:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
        message = str(e).lower()
        return "out of memory" in message or "cuda error" in message
