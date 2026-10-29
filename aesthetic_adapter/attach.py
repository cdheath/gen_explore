"""Registers the Aesthetic+Explore adapter onto a loaded FLUX transformer (Kontext or
FLUX.1-dev -- identical transformer config, confirmed this session, so this code is
shared between both pipeline stages; only the trained weights differ).

Only the 19 joint/double-stream blocks (`transformer_blocks.*`) get the adapter --
`FluxAestheticAttnProcessor` (and the stock class it forks) has no IP-adapter handling
for single-stream blocks by construction, see model.py's module docstring. The 38
`single_transformer_blocks.*` keep their existing (default `FluxAttnProcessor`) entries
untouched -- `transformer.set_attn_processor(...)` requires a dict covering every
processor, so those are passed through unchanged, not omitted.
"""

from __future__ import annotations

import torch.nn as nn

from .model import HIDDEN_SIZE, FluxAdapterProjector, FluxAestheticAttnProcessor, TwoSourceEncoderHidProj


class FluxAestheticAdapter(nn.Module):
    """Owns every trainable weight this adapter adds: the two source projectors
    (installed as `transformer.encoder_hid_proj`) and the 19 per-block processors
    (installed via `transformer.set_attn_processor`)."""

    def __init__(self, n_tokens: int = 4, hidden_size: int = HIDDEN_SIZE, num_blocks: int = 19):
        super().__init__()
        self.aesthetic_projector = FluxAdapterProjector(n_tokens=n_tokens, out_width=hidden_size)
        self.explore_projector = FluxAdapterProjector(n_tokens=n_tokens, out_width=hidden_size)
        self.encoder_hid_proj = TwoSourceEncoderHidProj(self.aesthetic_projector, self.explore_projector)
        self.processors = nn.ModuleList([
            FluxAestheticAttnProcessor(hidden_size=hidden_size, cross_attention_dim=hidden_size)
            for _ in range(num_blocks)
        ])


def attach(transformer, adapter: FluxAestheticAdapter) -> None:
    """Installs `adapter`'s modules onto `transformer` in place."""
    current = transformer.attn_processors
    joint_names = sorted(
        (n for n in current if n.startswith("transformer_blocks.")),
        key=lambda n: int(n.split(".")[1]),
    )
    if len(joint_names) != len(adapter.processors):
        raise ValueError(
            f"transformer has {len(joint_names)} joint-block attn processors but adapter "
            f"was built for {len(adapter.processors)} blocks."
        )

    new_processors = dict(current)  # single_transformer_blocks.* entries pass through untouched
    for name, proc in zip(joint_names, adapter.processors):
        new_processors[name] = proc
    transformer.set_attn_processor(new_processors)
    transformer.encoder_hid_proj = adapter.encoder_hid_proj
