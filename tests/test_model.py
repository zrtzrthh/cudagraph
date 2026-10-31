import torch

from cudagraph_decoder.model import ToyDecoder, ToyDecoderConfig


def test_toy_decoder_output_shape() -> None:
    model = ToyDecoder(
        ToyDecoderConfig(
            vocab_size=128,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            max_seq_len=16,
        )
    )
    input_ids = torch.randint(0, 128, (2, 8))
    output = model(input_ids)

    assert output.shape == (2, 8, 128)
