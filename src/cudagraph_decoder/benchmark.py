from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import torch

from .cudagraph import CUDAGraphDecoder
from .model import ToyDecoder, ToyDecoderConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int = 8
    seq_len: int = 64
    steps: int = 200
    warmup_steps: int = 20
    device: str = "cuda"
    vocab_size: int = 32000
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 128


def run_benchmark(config: BenchmarkConfig) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the benchmark.")

    device = torch.device(config.device)
    model = ToyDecoder(
        ToyDecoderConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
        )
    ).to(device)

    eager_inputs = [
        torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)
        for _ in range(config.steps)
    ]

    with torch.inference_mode():
        for _ in range(config.warmup_steps):
            model(eager_inputs[0])

    eager_latency = _measure_latency(lambda tensor: model(tensor), eager_inputs)

    graph_runner = CUDAGraphDecoder(model, device=device).capture(
        eager_inputs[0],
        warmup_steps=config.warmup_steps,
    )
    graph_latency = _measure_latency(lambda tensor: graph_runner.replay(tensor), eager_inputs)

    return {
        "eager_ms": eager_latency,
        "graph_ms": graph_latency,
        "speedup": eager_latency / graph_latency if graph_latency > 0 else float("inf"),
    }


def format_benchmark_result(config: BenchmarkConfig, result: Dict[str, float]) -> str:
    lines = ["CUDA Graph Decoder Benchmark"]
    lines.append(f"config: {asdict(config)}")
    lines.append(f"eager_ms: {result['eager_ms']:.4f}")
    lines.append(f"graph_ms: {result['graph_ms']:.4f}")
    lines.append(f"speedup: {result['speedup']:.2f}x")
    return "\n".join(lines)


def _measure_latency(fn: Callable[[torch.Tensor], torch.Tensor], inputs: List[torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for _ in range(5):
            fn(inputs[0])

        torch.cuda.synchronize()
        start.record()
        for tensor in inputs:
            fn(tensor)
        end.record()
        torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / max(len(inputs), 1)
