"""
Attention wrappers that record per-head post-o_proj contributions.

Llama: ported from external/JUICE/src/attention_head_utils_Llamma.py
  (GetHeadsLlamaAttention).
Gemma-2: ported from external/JUICE/src/attention_head_utils_Gemma.py
  (GetHeadsGemma2Attention), with dense_linear_weight_T for GPTQ.

Used by common.juice for the JuICE dual-run (save head outputs on pass 1,
inject scaled copies on pass 2). Do not use the in-forward
`intervention=True` path — that is JuICE's JRO ablation.
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
    softcap: Optional[float] = None,
    sliding_window=None,
    **kwargs,
):
    """Llama-identical matmul; Gemma-2 passes `softcap` (JuICE GetHeadsGemma2Attention).

    `sliding_window` is accepted so it is not lost when HF's SDPA/FA path is used;
    on eager, the decoder layer already baked the window into `attention_mask`
    (ConflictQA sequences are << 4096 anyway).
    """
    del sliding_window
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
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


def dense_linear_weight_T(linear: nn.Module) -> torch.Tensor:
    """
    Dense [in_features, out_features] matrix, same layout as `nn.Linear.weight.T`.

    FP16 `nn.Linear` has `.weight`. gptqmodel `TorchQuantLinear` does not; it exposes
    `dequantize_weight()` already in [in, out] (used as `x @ W` in forward).
    Prefer dequant when `qweight` is present so a leftover monkeypatched `.weight`
    cannot freeze the wrong dtype.
    """
    dequant = getattr(linear, "dequantize_weight", None)
    if callable(dequant) and getattr(linear, "qweight", None) is not None:
        in_f = int(getattr(linear, "in_features", 0) or 0)
        g_idx = getattr(linear, "g_idx", None)
        num_itr = 1
        if in_f and g_idx is not None and getattr(g_idx, "shape", [0])[0]:
            num_itr = max(1, int(g_idx.shape[0] // in_f))
        try:
            dw = dequant(num_itr=num_itr)
        except TypeError:
            dw = dequant()
        return dw.detach().contiguous()
    w = getattr(linear, "weight", None)
    if torch.is_tensor(w):
        return w.detach().T.contiguous()
    raise AttributeError(
        f"{type(linear).__name__} has no dense weight / dequantize_weight; "
        "cannot compute JuICE post-o_proj head contributions"
    )


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
        self._o_proj_wt: Optional[torch.Tensor] = None

    def _o_proj_weight_T(self, like: torch.Tensor) -> torch.Tensor:
        if self._o_proj_wt is None:
            self._o_proj_wt = dense_linear_weight_T(self.o_proj)
        wt = self._o_proj_wt
        if wt.device != like.device or wt.dtype != like.dtype:
            wt = wt.to(device=like.device, dtype=like.dtype)
            self._o_proj_wt = wt
        return wt

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
            weight_matrix = self._o_proj_weight_T(attn_output)
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


def is_gemma2_config(config) -> bool:
    mt = str(getattr(config, "model_type", "") or "").lower()
    if mt == "gemma2":
        return True
    arches = getattr(config, "architectures", None) or []
    return any("Gemma2" in str(a) for a in arches)


def juice_attn_class_for_config(config) -> type[nn.Module]:
    if is_gemma2_config(config):
        return GetHeadsGemma2Attention
    return GetHeadsLlamaAttention


def is_juice_wrapped_attn(attn: nn.Module) -> bool:
    return isinstance(attn, (GetHeadsLlamaAttention, GetHeadsGemma2Attention))


class GetHeadsGemma2Attention(nn.Module):
    """Port of JuICE GetHeadsGemma2Attention (post-o_proj head slices).

    Differences vs upstream `attention_head_utils_Gemma.py`:
    - `dense_linear_weight_T` instead of `o_proj.weight.T` (GPTQ has no `.weight`)
    - copies `sliding_window` from the native module when present (HF 4.51
      even/odd vs later `layer_types`); else the upstream even-layer rule
    """

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
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = original_self_attn.q_proj
        self.k_proj = original_self_attn.k_proj
        self.v_proj = original_self_attn.v_proj
        self.o_proj = original_self_attn.o_proj
        self._o_proj_wt: Optional[torch.Tensor] = None

        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        if hasattr(original_self_attn, "sliding_window"):
            self.sliding_window = original_self_attn.sliding_window
        else:
            self.sliding_window = (
                config.sliding_window if not bool(self.layer_idx % 2) else None
            )
        self.layer_type = getattr(original_self_attn, "layer_type", None)

    def _o_proj_weight_T(self, like: torch.Tensor) -> torch.Tensor:
        if self._o_proj_wt is None:
            self._o_proj_wt = dense_linear_weight_T(self.o_proj)
        wt = self._o_proj_wt
        if wt.device != like.device or wt.dtype != like.dtype:
            wt = wt.to(device=like.device, dtype=like.dtype)
            self._o_proj_wt = wt
        return wt

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

        past = past_key_value if past_key_value is not None else kwargs.pop("past_key_values", None)
        if past is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        kwargs.pop("sliding_window", None)
        kwargs.pop("softcap", None)

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
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if self.head_out:
            self.head_output = []
            weight_matrix = self._o_proj_weight_T(attn_output)
            for i in range(self.num_heads):
                input_attn = attn_output[:, :, i * self.head_dim : (i + 1) * self.head_dim]
                input_weight_matrix = weight_matrix[
                    i * self.head_dim : (i + 1) * self.head_dim, :
                ]
                self.head_output.append(input_attn @ input_weight_matrix)

        attn_output = self.o_proj(attn_output)

        if self.intervention and self.head_out:
            for h_i, alph in zip(self.h_I, self.alpha):
                attn_output = attn_output + self.head_output[h_i] * alph

        return attn_output, attn_weights

    def get_head_output(self):
        return self.head_output
