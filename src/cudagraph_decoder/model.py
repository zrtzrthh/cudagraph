from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class ToyDecoderConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.0


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden_states = self.norm1(hidden_states + attn_output)
        feed_forward_output = self.ffn(hidden_states)
        return self.norm2(hidden_states + feed_forward_output)


class ToyDecoder(nn.Module):
    """A small causal decoder used to demonstrate CUDA Graph capture."""

    def __init__(self, config: Optional[ToyDecoderConfig] = None) -> None:
        super().__init__()
        self.config = config or ToyDecoderConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.position_embedding = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        causal_mask = torch.triu(
            torch.ones(self.config.max_seq_len, self.config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len.")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)
