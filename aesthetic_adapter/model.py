"""Aesthetic + Explore cross-attention adapter for gen_explore's FLUX pipelines.

Ports the telepathy_egg_api/embedding_research SD3.5 aesthetic adapter (see
notes/aesthetic-cross-attention-spec.md in that repo) to FLUX -- but FLUX's own
IP-Adapter design (diffusers.models.transformers.transformer_flux) turns out to be
*natively* multi-source: `to_k_ip`/`to_v_ip` are already `nn.ModuleList`s (one entry per
conditioning source) and `ip_hidden_states` is already a `list[Tensor]`. That's a better
fit for "Aesthetic and Explore, independently toggleable" than the SD3.5 version needed
to hand-build -- two sources, one adapter, no per-combination retraining.

Three pieces:

  FluxAdapterProjector   256-d vector(s) -> n_tokens conditioning tokens, for ONE source.
                          No timestep argument needed here (unlike the SD3.5 version's
                          AdaLayerNorm modulation) -- confirmed against the real
                          FluxTransformer2DModel.forward(): `self.encoder_hid_proj(
                          ip_adapter_image_embeds)` is called with no timestep, simpler
                          than SD3's `image_proj(vectors, timestep)`.

  TwoSourceEncoderHidProj  Installed at `transformer.encoder_hid_proj`. Matches
                          `diffusers.models.embeddings.MultiIPAdapterImageProjection`'s
                          contract (`forward(image_embeds: list[Tensor]) -> list[Tensor]`,
                          a `num_ip_adapters` property) closely enough that
                          `FluxPipeline`/`FluxControlNetPipeline`'s own
                          `prepare_ip_adapter_image_embeds` works against it unmodified --
                          confirmed via source inspection, not assumed. Fixed at exactly
                          2 sources: index 0 = Aesthetic, index 1 = Explore.

  FluxAestheticAttnProcessor   Fork of
                          `diffusers.models.transformers.transformer_flux.FluxIPAdapterAttnProcessor`
                          (verified against the installed diffusers source this session).
                          Same math, same 3-tuple return contract
                          (`hidden_states, encoder_hidden_states, ip_attn_output` --
                          `FluxTransformerBlock.forward()` adds `ip_attn_output` itself,
                          the processor does NOT add it to hidden_states internally,
                          unlike the SD3.5 fork). The one real change: `self.scale`
                          (a plain `list[float]`, never trained) becomes
                          `self.gate = nn.ParameterList([...])`, one learned zero-init
                          scalar per source, so each source learns its own safe strength
                          per block instead of one hand-set constant -- the same fix the
                          SD3.5 M3 adapter made, now per-source as well as per-block.

Scope note (confirmed structurally, not just as a memory-saving choice): the stock
`FluxIPAdapterAttnProcessor.__call__`'s `else` branch (reached when
`encoder_hidden_states is None`, i.e. FLUX's single-stream blocks, which operate on an
already-merged token stream) returns plain `hidden_states` with **no IP-adapter handling
at all** -- this mechanism is only wired for the 19 joint/double-stream blocks by design,
before any deliberate scoping decision on top. attach.py only touches those 19 blocks.

Requirement carried over from the SD3.5 build's own fixed bug: `ip_hidden_states` must
never be omitted when this processor is attached -- the loop over
`zip(ip_hidden_states, self.gate, self.to_k_ip, self.to_v_ip)` assumes a real list, and
skipping it (rather than passing zero vectors) removes every adapter parameter from that
step's autograd graph, which crashed `loss.backward()` during the SD3.5 training loop.
Always pass real-or-zero tensors for both sources, on every call.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

HIDDEN_SIZE = 3072  # FLUX: attention_head_dim(128) * num_attention_heads(24)
HEAD_DIM = 128
N_TOKENS = 4
NUM_SOURCES = 2  # 0 = aesthetic, 1 = explore -- fixed, not configurable per instance
SOURCE_NAMES = ("aesthetic", "explore")


class FluxAdapterProjector(nn.Module):
    """256 -> n_tokens * hidden_size, reshaped to [B, n_tokens, hidden_size] per input
    vector, for one conditioning source. Same shape/design as the SD3.5 adapter's
    AestheticProjector; no timestep submodule needed for FLUX."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 1024,
                 n_tokens: int = N_TOKENS, out_width: int = HIDDEN_SIZE):
        super().__init__()
        self.n_tokens = n_tokens
        self.out_width = out_width
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_tokens * out_width),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """vectors: [N, 256] (N = B * n_refs, already flattened by the caller) ->
        [N, n_tokens, out_width]."""
        out = self.net(vectors)
        return out.view(vectors.shape[0], self.n_tokens, self.out_width)


class TwoSourceEncoderHidProj(nn.Module):
    """Installed at `transformer.encoder_hid_proj`. `image_embeds` is a list of exactly
    2 tensors, each `[B, n_refs, 256]` (n_refs=1 for a single/averaged reference,
    n_refs>1 keeps references separate -- same average/multi_token distinction the
    SD3.5 adapter used, now per-source). Returns a list of 2 `[B, n_refs*n_tokens,
    hidden_size]` token tensors, matching `MultiIPAdapterImageProjection`'s contract."""

    def __init__(self, aesthetic_projector: FluxAdapterProjector, explore_projector: FluxAdapterProjector):
        super().__init__()
        self.image_projection_layers = nn.ModuleList([aesthetic_projector, explore_projector])

    @property
    def num_ip_adapters(self) -> int:
        return len(self.image_projection_layers)

    def forward(self, image_embeds: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(image_embeds) != self.num_ip_adapters:
            raise ValueError(f"Expected {self.num_ip_adapters} conditioning sources, got {len(image_embeds)}.")
        outputs = []
        for embeds, proj in zip(image_embeds, self.image_projection_layers):
            b, n_refs, d = embeds.shape
            flat = embeds.reshape(b * n_refs, d)
            tokens = proj(flat.to(proj.net[0].weight.dtype))
            outputs.append(tokens.view(b, n_refs * proj.n_tokens, proj.out_width))
        return outputs


class FluxAestheticAttnProcessor(nn.Module):
    """Fork of diffusers' FluxIPAdapterAttnProcessor -- see module docstring for the one
    real change (learned per-source, per-block gate instead of a fixed list of floats).
    Only valid for joint/double-stream blocks (encoder_hidden_states is not None)."""

    def __init__(self, hidden_size: int, cross_attention_dim: int, num_tokens=(N_TOKENS, N_TOKENS)):
        super().__init__()
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.gate = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in num_tokens])
        self.to_k_ip = nn.ModuleList([
            nn.Linear(cross_attention_dim, hidden_size, bias=True) for _ in num_tokens
        ])
        self.to_v_ip = nn.ModuleList([
            nn.Linear(cross_attention_dim, hidden_size, bias=True) for _ in num_tokens
        ])

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        ip_hidden_states: list[torch.Tensor] | None = None,
        ip_adapter_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from diffusers.models.transformers.transformer_flux import _get_qkv_projections
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        batch_size = hidden_states.shape[0]

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        ip_query = query

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            ip_attn_output = torch.zeros_like(hidden_states)

            for current_ip_hidden_states, gate, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.gate, self.to_k_ip, self.to_v_ip
            ):
                adapter_dtype = to_k_ip.weight.dtype
                ip_key = to_k_ip(current_ip_hidden_states.to(adapter_dtype))
                ip_value = to_v_ip(current_ip_hidden_states.to(adapter_dtype))

                ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                current_ip_hidden_states = dispatch_attention_fn(
                    ip_query.to(adapter_dtype), ip_key, ip_value,
                    attn_mask=None, dropout_p=0.0, is_causal=False,
                )
                current_ip_hidden_states = current_ip_hidden_states.reshape(batch_size, -1, attn.heads * attn.head_dim)
                ip_attn_output = ip_attn_output + (gate * current_ip_hidden_states).to(hidden_states.dtype)

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            # Single-stream blocks (no encoder_hidden_states) -- not conditioned, see
            # module docstring. This matches the stock class's own behavior exactly.
            return hidden_states
