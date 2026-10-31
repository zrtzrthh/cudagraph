import argparse
import sys
from typing import List, Optional

from .benchmark import BenchmarkConfig, format_benchmark_result, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark decoder inference with CUDA Graphs.")
    parser.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=BenchmarkConfig.seq_len)
    parser.add_argument("--steps", type=int, default=BenchmarkConfig.steps)
    parser.add_argument("--warmup-steps", type=int, default=BenchmarkConfig.warmup_steps)
    parser.add_argument("--vocab-size", type=int, default=BenchmarkConfig.vocab_size)
    parser.add_argument("--hidden-size", type=int, default=BenchmarkConfig.hidden_size)
    parser.add_argument("--num-heads", type=int, default=BenchmarkConfig.num_heads)
    parser.add_argument("--num-layers", type=int, default=BenchmarkConfig.num_layers)
    parser.add_argument("--max-seq-len", type=int, default=BenchmarkConfig.max_seq_len)
    parser.add_argument("--device", type=str, default=BenchmarkConfig.device)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        device=args.device,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )

    try:
        result = run_benchmark(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_benchmark_result(config, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
