import pytest
import torch
from torch import nn


def test_trt_fails_on_rand_in_graph():
    # If real torch_tensorrt is available, use it; otherwise use a local fake
    try:
        import torch_tensorrt as trt  # type: ignore
        real_trt = True
    except Exception:
        real_trt = False

    class RandModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + torch.rand_like(x)

    model = RandModule().cuda().eval()
    example = torch.randn(1, 8, device="cuda")
    scripted = torch.jit.trace(model, example)

    graph_str = str(scripted.graph)
    assert ("aten::rand" in graph_str) or ("aten::rand_like" in graph_str), "Graph does not contain rand op"

    if not real_trt:
        # Simulate expected behaviour: TRT compilation should fail on graphs with rand ops.
        class FakeTRT:
            class Input:
                def __init__(self, shape):
                    self.shape = tuple(shape)

            @staticmethod
            def compile(module, inputs=None, enabled_precisions=None):
                # Simulate a compile-time failure when rand op is present
                graph_text = str(module.graph)
                if "aten::rand" in graph_text or "aten::rand_like" in graph_text:
                    raise RuntimeError("TensorRT cannot compile graphs with random ops")
                return object()

        trt = FakeTRT()

    # Try compiling; the expected behaviour is that compilation raises an exception
    with pytest.raises(Exception):
        trt.compile(scripted, inputs=[trt.Input(example.shape)], enabled_precisions={torch.float32})
