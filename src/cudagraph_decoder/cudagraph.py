from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class CUDAGraphConfig:
    warmup_steps: int = 3


class CUDAGraphDecoder:
    """Capture and replay a fixed-shape decoder forward pass with CUDA Graphs."""

    def __init__(self, module: nn.Module, device: Optional[torch.device] = None) -> None:
        if device is None:
            try:
                device = next(module.parameters()).device
            except StopIteration as exc:
                raise ValueError("The module must have parameters or an explicit device") from exc

        self.device = torch.device(device)
        self.module = module.to(self.device).eval()
        self.graph = torch.cuda.CUDAGraph()
        self._static_inputs: Optional[List[torch.Tensor]] = None
        self._static_outputs: Any = None

    def capture(self, *example_inputs: torch.Tensor, warmup_steps: int = 3) -> "CUDAGraphDecoder":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDAGraphDecoder.capture()")

        static_inputs = [self._make_static_tensor(tensor) for tensor in example_inputs]
        warmup_stream = torch.cuda.Stream()

        with torch.inference_mode():
            with torch.cuda.stream(warmup_stream):
                for _ in range(warmup_steps):
                    self.module(*static_inputs)
            torch.cuda.synchronize()

            with torch.cuda.graph(self.graph):
                self._static_outputs = self.module(*static_inputs)

        self._static_inputs = static_inputs
        return self

    def replay(self, *inputs: torch.Tensor) -> Any:
        if self._static_inputs is None:
            raise RuntimeError("Call capture() before replay().")
        if len(inputs) != len(self._static_inputs):
            raise ValueError("Input count does not match the captured graph.")

        with torch.inference_mode():
            for static_tensor, new_tensor in zip(self._static_inputs, inputs):
                if static_tensor.shape != new_tensor.shape:
                    raise ValueError("Input shape does not match the captured graph.")
                if static_tensor.dtype != new_tensor.dtype:
                    raise ValueError("Input dtype does not match the captured graph.")
                static_tensor.copy_(new_tensor, non_blocking=True)

            self.graph.replay()

        return self._static_outputs

    def _make_static_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cpu":
            return tensor.to(self.device, non_blocking=True).clone()
        return tensor.detach().clone().to(self.device)
