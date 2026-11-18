"""
Llama attention wrapper that records per-head post-o_proj contributions.

Ported from external/JUICE/src/attention_head_utils_Llamma.py
(GetHeadsLlamaAttention + helpers). Used by common.juice for the real
JuICE dual-run (save head outputs on pass 1, inject scaled copies on pass 2).

Do not use the in-forward `intervention=True` path for JuICE — that is their
"just run once" (JRO) ablation. JuICE / JRT injects *cached* pass-1 outputs.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GetHeadsLlamaAttention(nn.Module):
    """Exact port of JuICE GetHeadsLlamaAttention (post-o_proj head slices)."""

    def __init__(
        self,
        original_self_attn,
        h_I=None,
        alpha=None,
        intervention=False,
        head_out=True,
    ):
        super().__init__()
        config = original_self_attn.config
        self.config = original_self_attn.config
        self.layer_idx = original_self_attn.layer_idx

        self.h_I = h_I if h_I is not None else []
        self.alpha = alpha if alpha is not None else []
        self.intervention = intervention
        self.head_out = head_out
        self.head_output = []
        self.num_heads = config.num_attention_heads

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = original_self_attn.q_proj
        self.k_proj = original_self_attn.k_proj
        self.v_proj = original_self_attn.v_proj
        self.o_proj = original_self_attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                pass
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if self.head_out:
            # Per-head contribution after the o_proj linear (JuICE definition).
            self.head_output = []
            weight_matrix = self.o_proj.weight.detach().T
            for i in range(self.num_heads):
                input_attn = attn_output[:, :, i * self.head_dim : (i + 1) * self.head_dim]
                input_weight_matrix = weight_matrix[
                    i * self.head_dim : (i + 1) * self.head_dim, :
                ]
                self.head_output.append(input_attn @ input_weight_matrix)

        attn_output = self.o_proj(attn_output)

        # JRO path only — JuICE dual-run must NOT rely on this.
        if self.intervention and self.head_out:
            for h_i, alph in zip(self.h_I, self.alpha):
                attn_output = attn_output + self.head_output[h_i] * alph

        return attn_output, attn_weights

    def get_head_output(self):
        return self.head_output
