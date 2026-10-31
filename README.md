# CUDA Graph Decoder Demo

This project shows how to wrap a decoder-style PyTorch module with CUDA Graph capture so repeated inference on fixed shapes can avoid Python overhead and reduce launch latency.

## What is included

- `ToyDecoder`: a small causal decoder built with PyTorch modules.
- `CUDAGraphDecoder`: a reusable wrapper that captures and replays a fixed-shape decoder step.
- `cudagraph-decoder-bench`: a CLI benchmark that compares eager execution against CUDA Graph replay.

## Quick start

1. Install a CUDA-enabled PyTorch build that matches your system.
2. Install the package in editable mode:

```bash
pip install -e .
```

3. Run the benchmark on a GPU machine:

```bash
cudagraph-decoder-bench --batch-size 8 --seq-len 64 --steps 200
```

If CUDA is not available, the benchmark prints a clear error message and exits.

## Project layout

- `src/cudagraph_decoder/model.py`: the decoder model used for the demo.
- `src/cudagraph_decoder/cudagraph.py`: the CUDA Graph capture/replay wrapper.
- `src/cudagraph_decoder/benchmark.py`: eager vs CUDA Graph timing.
- `src/cudagraph_decoder/cli.py`: command line entry point.

## Notes

CUDA Graph capture requires fixed tensor shapes, device placement, and execution paths. This project keeps the example focused on a decoder step so it is easy to adapt to real autoregressive inference code.
