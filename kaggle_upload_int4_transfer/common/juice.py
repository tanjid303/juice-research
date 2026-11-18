"""
JuICE (Just Run Twice) — faithful dual-run head ID + steering for Llama.

Ported from GaotangLi/JUICE (ICML 2025), NOT the earlier prefill-only approximation.

Source mapping (external/JUICE):
- Head contribution: attention_head_utils_Llamma.GetHeadsLlamaAttention
  (post-o_proj per-head: attn_slice @ W_o_slice)
- Identification dual-run: select_intervention_head.py
  (pass 1 record head_output; pass 2 o_proj.output += alpha * saved)
- Generation dual-run: inference_utils.generate_with_intervention
  (two forwards *per decode step*; never reuse a prefill-only cache)

Differences vs the old common/juice.py approximation:
1. Caches post-o_proj head *contribution* (hidden_size), not pre-o_proj head slice.
2. Injects onto o_proj *output*, not into o_proj input.
3. Generation: true dual-run every token; does not reuse prefill activations in decode.
4. Head scoring: ΔP(first target token) averaged over JuICE alphas, with H+/H− sets.

K: paper/code use K=10 for Llama2/Llama3/Olmo (K=5 for smaller models).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from common.juice_attn import GetHeadsLlamaAttention
from common.models import input_device
from common.prompts import ConflictItem, apply_chat_template, juice_style_factual_prompts
from common.scoring import label_conflict_response

SteerTarget = Literal["context", "parametric"]

# JuICE select_intervention_head.py defaults (slightly trimmed neg set for VRAM/time).
DEFAULT_POS_ALPHAS: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 30.0)
DEFAULT_NEG_ALPHAS: tuple[float, ...] = (-1.0, -2.0, -5.0)
# Practical Kaggle ID defaults (same signs/roles; fewer magnitudes).
FAST_POS_ALPHAS: tuple[float, ...] = (1.0, 3.0, 5.0)
FAST_NEG_ALPHAS: tuple[float, ...] = (-1.0, -2.0)


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int

    def key(self) -> str:
        return f"L{self.layer}_H{self.head}"

    def as_tuple(self) -> tuple[int, int]:
        return (self.layer, self.head)


@dataclass
class IdentifiedHeads:
    """H+ (enlarge) and H− (suppress) for one steer target, JuICE-style."""

    enlarge: list[HeadRef]
    suppress: list[HeadRef]
    scores_pos: dict[str, float]
    scores_neg: dict[str, float]
    precision: str
    K: int
    steer_target: SteerTarget
    n_calibration: int
    pos_alphas: list[float]
    neg_alphas: list[float]

    @property
    def heads(self) -> list[HeadRef]:
        """Back-compat: enlarge heads first."""
        return list(self.enlarge)

    def to_json(self) -> dict:
        return {
            "enlarge": [asdict(h) for h in self.enlarge],
            "suppress": [asdict(h) for h in self.suppress],
            "scores_pos": self.scores_pos,
            "scores_neg": self.scores_neg,
            "precision": self.precision,
            "K": self.K,
            "steer_target": self.steer_target,
            "n_calibration": self.n_calibration,
            "pos_alphas": self.pos_alphas,
            "neg_alphas": self.neg_alphas,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "IdentifiedHeads":
        if "enlarge" in obj:
            enlarge = [HeadRef(**h) for h in obj["enlarge"]]
            suppress = [HeadRef(**h) for h in obj.get("suppress", [])]
            return cls(
                enlarge=enlarge,
                suppress=suppress,
                scores_pos=obj.get("scores_pos", obj.get("scores", {})),
                scores_neg=obj.get("scores_neg", {}),
                precision=obj["precision"],
                K=obj["K"],
                steer_target=obj["steer_target"],
                n_calibration=obj["n_calibration"],
                pos_alphas=list(obj.get("pos_alphas", [])),
                neg_alphas=list(obj.get("neg_alphas", [])),
            )
        # Legacy single-list format from the approximation era.
        heads = [HeadRef(**h) for h in obj["heads"]]
        return cls(
            enlarge=heads,
            suppress=[],
            scores_pos=obj.get("scores", {}),
            scores_neg={},
            precision=obj["precision"],
            K=obj["K"],
            steer_target=obj["steer_target"],
            n_calibration=obj["n_calibration"],
            pos_alphas=[],
            neg_alphas=[],
        )


def save_identified_heads(path: str | Path, ident: IdentifiedHeads) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(ident.to_json(), f, indent=2)


def load_identified_heads(path: str | Path) -> IdentifiedHeads:
    with Path(path).open("r", encoding="utf-8") as f:
        return IdentifiedHeads.from_json(json.load(f))


def model_dims(model: PreTrainedModel) -> tuple[int, int, int]:
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    return n_layers, n_heads, head_dim


def wrap_llama_attention_for_juice(model: PreTrainedModel) -> None:
    """Replace each layer's self_attn with GetHeadsLlamaAttention (shares proj modules)."""
    import torch

    for layer in model.model.layers:
        if isinstance(layer.self_attn, GetHeadsLlamaAttention):
            continue
        layer.self_attn = GetHeadsLlamaAttention(
            layer.self_attn,
            h_I=[],
            alpha=[],
            intervention=False,
            head_out=False,
        )

    # Cache dequantized o_proj W once (GPTQ has no .weight). Cast happens per forward.
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for p in model.parameters():
        if p.dtype in (torch.float16, torch.bfloat16):
            dtype = p.dtype
            device = p.device
            break
    n = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        if not isinstance(attn, GetHeadsLlamaAttention):
            continue
        if getattr(attn, "_o_proj_wt", None) is not None:
            continue
        o = attn.o_proj
        scales = getattr(o, "scales", None)
        if torch.is_tensor(scales):
            device = scales.device
            if scales.dtype in (torch.float16, torch.bfloat16):
                dtype = scales.dtype
        attn._o_proj_weight_T(torch.empty(0, device=device, dtype=dtype))
        n += 1
    print(f"JuICE: cached o_proj W on {n} layers ({dtype})")


def _set_head_out(model: PreTrainedModel, enabled: bool) -> None:
    for layer in model.model.layers:
        attn = layer.self_attn
        if isinstance(attn, GetHeadsLlamaAttention):
            attn.head_out = enabled
            attn.intervention = False
            if not enabled:
                attn.head_output = []


def _first_token_id(tokenizer: PreTrainedTokenizerBase, text: str) -> int | None:
    """Match JuICE get_tokenized_first_word + get_probability_of_word for Llama-3."""
    word = text.strip().split()[0] if text.strip() else ""
    if not word:
        return None
    # Llama-3 / Gemma path in their inference_utils: leading space on first word.
    spaced = f" {word}"
    ids = tokenizer.encode(spaced, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def _token_prob_from_logits(
    logits_last: torch.Tensor,
    token_id: int | None,
) -> float:
    if token_id is None:
        return 0.0
    probs = torch.softmax(logits_last.float(), dim=-1)
    return float(probs[token_id].item())


def _tokenize_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> dict[str, torch.Tensor]:
    dev = input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(dev) for k, v in enc.items()}


@torch.inference_mode()
def _forward_logits(model: PreTrainedModel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs, use_cache=False)
    return out.logits[0, -1, :]


def _collect_all_head_outputs(
    model: PreTrainedModel,
) -> dict[tuple[int, int], torch.Tensor]:
    """After a head_out=True forward, snapshot every (layer, head) contribution."""
    cached: dict[tuple[int, int], torch.Tensor] = {}
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if not isinstance(attn, GetHeadsLlamaAttention) or not attn.head_output:
            continue
        for hi, tens in enumerate(attn.head_output):
            cached[(li, hi)] = tens.detach()
    return cached


def _inject_o_proj_hooks(
    model: PreTrainedModel,
    specs: Sequence[tuple[int, int, float, torch.Tensor]],
) -> list:
    """
    Register forward hooks: o_proj.output += alpha * saved_head_out
    (matches nnsight `o_proj.output += alpha * head_outs[i]` in JuICE).
    """
    by_layer: dict[int, list[tuple[float, torch.Tensor]]] = defaultdict(list)
    for layer, _head, alpha, saved in specs:
        by_layer[layer].append((alpha, saved))

    handles = []
    for layer, items in by_layer.items():
        o_proj = model.model.layers[layer].self_attn.o_proj
        # Bind items in default-arg to avoid late-binding closure bugs.
        def _hook(_module, _args, output, _items=items):
            out = output
            for alpha, saved in _items:
                out = out + alpha * saved.to(device=out.device, dtype=out.dtype)
            return out

        handles.append(o_proj.register_forward_hook(_hook))
    return handles


@torch.inference_mode()
def dual_run_logits(
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor],
    interventions: Sequence[tuple[int, int, float]],
    *,
    cached_heads: dict[tuple[int, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    JuICE dual-run for a single forward (logit readout at last position).

    interventions: list of (layer, head, alpha)
    If cached_heads is None, run pass 1 with head_out to record them.
    """
    wrap_llama_attention_for_juice(model)

    if cached_heads is None:
        _set_head_out(model, True)
        _ = model(**inputs, use_cache=False)
        cached_heads = _collect_all_head_outputs(model)
        _set_head_out(model, False)

    specs = []
    for layer, head, alpha in interventions:
        key = (layer, head)
        if key not in cached_heads:
            raise KeyError(f"Missing cached head output for {key}")
        specs.append((layer, head, alpha, cached_heads[key]))

    handles = _inject_o_proj_hooks(model, specs)
    try:
        _set_head_out(model, False)
        logits = _forward_logits(model, inputs)
    finally:
        for h in handles:
            h.remove()
    return logits


def identify_reliable_heads(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_items: Sequence[ConflictItem],
    *,
    precision: str,
    K: int = 10,
    steer_target: SteerTarget = "context",
    pos_alphas: Sequence[float] = FAST_POS_ALPHAS,
    neg_alphas: Sequence[float] = FAST_NEG_ALPHAS,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
) -> IdentifiedHeads:
    """
    Rank heads by mean ΔP(target_token) under dual-run ±alpha injection.

    Scores both H+ (positive alphas) and H− (negative alphas) like
    select_intervention_head.py. Target token = context or parametric answer.
    """
    random.seed(seed)
    wrap_llama_attention_for_juice(model)
    n_layers, n_heads, _ = model_dims(model)
    pos_alphas = list(pos_alphas)
    neg_alphas = list(neg_alphas)
    all_alphas = pos_alphas + neg_alphas

    # Accumulators: head_key -> list of per-(item,alpha) ΔP
    pos_shifts: dict[str, list[float]] = defaultdict(list)
    neg_shifts: dict[str, list[float]] = defaultdict(list)

    prompts = [
        apply_chat_template(tokenizer, it, "context_conflicts") for it in calibration_items
    ]

    for item, prompt in tqdm(
        list(zip(calibration_items, prompts)),
        desc=f"juice_id[{steer_target}]",
    ):
        target_text = (
            item.context_answer if steer_target == "context" else item.parametric_answer
        )
        tid = _first_token_id(tokenizer, target_text)
        inputs = _tokenize_prompt(model, tokenizer, prompt)

        # Pass 1 once per item: record all head contributions.
        _set_head_out(model, True)
        clean_logits = _forward_logits(model, inputs)
        cached = _collect_all_head_outputs(model)
        _set_head_out(model, False)
        p0 = _token_prob_from_logits(clean_logits, tid)

        for layer in range(n_layers):
            for head in range(n_heads):
                href = HeadRef(layer, head)
                for alpha in all_alphas:
                    try:
                        logits = dual_run_logits(
                            model,
                            inputs,
                            [(layer, head, alpha)],
                            cached_heads=cached,
                        )
                        p1 = _token_prob_from_logits(logits, tid)
                        delta = p1 - p0
                    except RuntimeError:
                        delta = 0.0
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if alpha > 0:
                        pos_shifts[href.key()].append(delta)
                    else:
                        neg_shifts[href.key()].append(delta)

        if checkpoint_path is not None:
            _dump_partial_scores(checkpoint_path, pos_shifts, neg_shifts, steer_target)

        # Free per-item activation cache.
        del cached
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores_pos = {k: (sum(v) / len(v) if v else 0.0) for k, v in pos_shifts.items()}
    scores_neg = {k: (sum(v) / len(v) if v else 0.0) for k, v in neg_shifts.items()}

    # JuICE keep_* : only heads with consistent positive mean effect.
    pos_pool = {k: s for k, s in scores_pos.items() if s > 0}
    neg_pool = {k: s for k, s in scores_neg.items() if s > 0}

    enlarge = _top_k_heads(pos_pool, K)
    suppress = _top_k_heads(neg_pool, K)
    # Avoid double-applying the same head in both sets.
    enlarge_keys = {h.key() for h in enlarge}
    suppress = [h for h in suppress if h.key() not in enlarge_keys][:K]

    return IdentifiedHeads(
        enlarge=enlarge,
        suppress=suppress,
        scores_pos=dict(sorted(scores_pos.items(), key=lambda x: -x[1])),
        scores_neg=dict(sorted(scores_neg.items(), key=lambda x: -x[1])),
        precision=precision,
        K=K,
        steer_target=steer_target,
        n_calibration=len(calibration_items),
        pos_alphas=pos_alphas,
        neg_alphas=neg_alphas,
    )


def identify_both_directions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_items: Sequence[ConflictItem],
    *,
    precision: str,
    K: int = 10,
    pos_alphas: Sequence[float] = FAST_POS_ALPHAS,
    neg_alphas: Sequence[float] = FAST_NEG_ALPHAS,
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
) -> tuple[IdentifiedHeads, IdentifiedHeads]:
    """
    One dual-run sweep scoring ΔP(context) and ΔP(parametric) from the same logits.

    Much cheaper than calling identify_reliable_heads twice.
    """
    random.seed(seed)
    wrap_llama_attention_for_juice(model)
    n_layers, n_heads, _ = model_dims(model)
    pos_alphas = list(pos_alphas)
    neg_alphas = list(neg_alphas)
    all_alphas = pos_alphas + neg_alphas

    pos_ctx: dict[str, list[float]] = defaultdict(list)
    neg_ctx: dict[str, list[float]] = defaultdict(list)
    pos_par: dict[str, list[float]] = defaultdict(list)
    neg_par: dict[str, list[float]] = defaultdict(list)

    prompts = [
        apply_chat_template(tokenizer, it, "context_conflicts") for it in calibration_items
    ]
    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)

    for item, prompt in tqdm(
        list(zip(calibration_items, prompts)),
        desc="juice_id[both]",
    ):
        tid_c = _first_token_id(tokenizer, item.context_answer)
        tid_p = _first_token_id(tokenizer, item.parametric_answer)
        inputs = _tokenize_prompt(model, tokenizer, prompt)

        _set_head_out(model, True)
        clean_logits = _forward_logits(model, inputs)
        cached = _collect_all_head_outputs(model)
        _set_head_out(model, False)
        p0_c = _token_prob_from_logits(clean_logits, tid_c)
        p0_p = _token_prob_from_logits(clean_logits, tid_p)

        for layer in range(n_layers):
            for head in range(n_heads):
                key = HeadRef(layer, head).key()
                for alpha in all_alphas:
                    try:
                        logits = dual_run_logits(
                            model,
                            inputs,
                            [(layer, head, alpha)],
                            cached_heads=cached,
                        )
                        d_c = _token_prob_from_logits(logits, tid_c) - p0_c
                        d_p = _token_prob_from_logits(logits, tid_p) - p0_p
                    except RuntimeError:
                        d_c = d_p = 0.0
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if alpha > 0:
                        pos_ctx[key].append(d_c)
                        pos_par[key].append(d_p)
                    else:
                        neg_ctx[key].append(d_c)
                        neg_par[key].append(d_p)

        if ckpt is not None:
            _dump_partial_scores(ckpt / "partial_context.json", pos_ctx, neg_ctx, "context")
            _dump_partial_scores(ckpt / "partial_parametric.json", pos_par, neg_par, "parametric")

        del cached
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ctx = _finalize_identified(
        pos_ctx, neg_ctx, precision=precision, K=K, steer_target="context",
        n_calibration=len(calibration_items), pos_alphas=pos_alphas, neg_alphas=neg_alphas,
    )
    par = _finalize_identified(
        pos_par, neg_par, precision=precision, K=K, steer_target="parametric",
        n_calibration=len(calibration_items), pos_alphas=pos_alphas, neg_alphas=neg_alphas,
    )
    return ctx, par


def identify_both_directions_juice_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_items: Sequence[ConflictItem],
    *,
    precision: str,
    K: int = 10,
    pos_alphas: Sequence[float] = FAST_POS_ALPHAS,
    neg_alphas: Sequence[float] = FAST_NEG_ALPHAS,
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
    require_all_prompt_types_positive: bool = True,
) -> tuple[IdentifiedHeads, IdentifiedHeads, dict]:
    """
    JuICE-faithful ID for base models: plain factual prompts, 3 types per item
    (clean / substitution_conflict / coherent_conflict), no chat template.

    Returns (heads_ctx, heads_par, diagnostics_blob) where diagnostics_blob includes
    per-prompt-type means for consistency checks and Instruct comparison.
    """
    random.seed(seed)
    wrap_llama_attention_for_juice(model)
    n_layers, n_heads, _ = model_dims(model)
    pos_alphas = list(pos_alphas)
    neg_alphas = list(neg_alphas)
    all_alphas = pos_alphas + neg_alphas
    prompt_types = ("clean", "substitution_conflict", "coherent_conflict")

    # overall accumulators (Instruct-comparable)
    pos_ctx: dict[str, list[float]] = defaultdict(list)
    neg_ctx: dict[str, list[float]] = defaultdict(list)
    pos_par: dict[str, list[float]] = defaultdict(list)
    neg_par: dict[str, list[float]] = defaultdict(list)
    # per prompt-type means building blocks: key -> ptype -> list
    pos_ctx_pt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pos_par_pt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)

    for item in tqdm(list(calibration_items), desc="juice_id[base,3ptype]"):
        prompts = juice_style_factual_prompts(item)
        tid_c = _first_token_id(tokenizer, item.context_answer)
        tid_p = _first_token_id(tokenizer, item.parametric_answer)

        for ptype in prompt_types:
            prompt = prompts[ptype]
            inputs = _tokenize_prompt(model, tokenizer, prompt)

            _set_head_out(model, True)
            clean_logits = _forward_logits(model, inputs)
            cached = _collect_all_head_outputs(model)
            _set_head_out(model, False)
            p0_c = _token_prob_from_logits(clean_logits, tid_c)
            p0_p = _token_prob_from_logits(clean_logits, tid_p)

            for layer in range(n_layers):
                for head in range(n_heads):
                    key = HeadRef(layer, head).key()
                    for alpha in all_alphas:
                        try:
                            logits = dual_run_logits(
                                model,
                                inputs,
                                [(layer, head, alpha)],
                                cached_heads=cached,
                            )
                            d_c = _token_prob_from_logits(logits, tid_c) - p0_c
                            d_p = _token_prob_from_logits(logits, tid_p) - p0_p
                        except RuntimeError:
                            d_c = d_p = 0.0
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        if alpha > 0:
                            pos_ctx[key].append(d_c)
                            pos_par[key].append(d_p)
                            pos_ctx_pt[key][ptype].append(d_c)
                            pos_par_pt[key][ptype].append(d_p)
                        else:
                            neg_ctx[key].append(d_c)
                            neg_par[key].append(d_p)

            del cached
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if ckpt is not None:
            _dump_partial_scores(ckpt / "partial_context.json", pos_ctx, neg_ctx, "context")
            _dump_partial_scores(ckpt / "partial_parametric.json", pos_par, neg_par, "parametric")

    def _ptype_means(pt_map: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, by_pt in pt_map.items():
            out[key] = {
                pt: (sum(vs) / len(vs) if vs else 0.0) for pt, vs in by_pt.items()
            }
        return out

    ctx_pt_means = _ptype_means(pos_ctx_pt)
    par_pt_means = _ptype_means(pos_par_pt)

    ctx = _finalize_identified(
        pos_ctx,
        neg_ctx,
        precision=precision,
        K=K,
        steer_target="context",
        n_calibration=len(calibration_items),
        pos_alphas=pos_alphas,
        neg_alphas=neg_alphas,
        require_prompt_type_means=ctx_pt_means if require_all_prompt_types_positive else None,
        prompt_types=prompt_types,
    )
    par = _finalize_identified(
        pos_par,
        neg_par,
        precision=precision,
        K=K,
        steer_target="parametric",
        n_calibration=len(calibration_items),
        pos_alphas=pos_alphas,
        neg_alphas=neg_alphas,
        require_prompt_type_means=par_pt_means if require_all_prompt_types_positive else None,
        prompt_types=prompt_types,
    )

    diagnostics = {
        "n_calibration": len(calibration_items),
        "prompt_types": list(prompt_types),
        "require_all_prompt_types_positive": require_all_prompt_types_positive,
        "scores_pos_context_by_prompt_type": ctx_pt_means,
        "scores_pos_parametric_by_prompt_type": par_pt_means,
        "scores_pos_context": ctx.scores_pos,
        "scores_pos_parametric": par.scores_pos,
        "scores_neg_context": ctx.scores_neg,
        "scores_neg_parametric": par.scores_neg,
    }
    return ctx, par, diagnostics


def _finalize_identified(
    pos_shifts: dict[str, list[float]],
    neg_shifts: dict[str, list[float]],
    *,
    precision: str,
    K: int,
    steer_target: SteerTarget,
    n_calibration: int,
    pos_alphas: list[float],
    neg_alphas: list[float],
    require_prompt_type_means: dict[str, dict[str, float]] | None = None,
    prompt_types: Sequence[str] = (),
) -> IdentifiedHeads:
    scores_pos = {k: (sum(v) / len(v) if v else 0.0) for k, v in pos_shifts.items()}
    scores_neg = {k: (sum(v) / len(v) if v else 0.0) for k, v in neg_shifts.items()}
    pos_pool = {k: s for k, s in scores_pos.items() if s > 0}
    neg_pool = {k: s for k, s in scores_neg.items() if s > 0}

    # JuICE-style: keep only heads with positive mean on every prompt type.
    if require_prompt_type_means is not None and prompt_types:
        filtered = {}
        for k, s in pos_pool.items():
            means = require_prompt_type_means.get(k, {})
            if all(means.get(pt, 0.0) > 0 for pt in prompt_types):
                # Rank by sum across prompt types (their total_pos_increase).
                filtered[k] = sum(means.get(pt, 0.0) for pt in prompt_types)
        if filtered:
            pos_pool = filtered

    enlarge = _top_k_heads(pos_pool, K)
    enlarge_keys = {h.key() for h in enlarge}
    suppress = [h for h in _top_k_heads(neg_pool, K) if h.key() not in enlarge_keys][:K]
    return IdentifiedHeads(
        enlarge=enlarge,
        suppress=suppress,
        scores_pos=dict(sorted(scores_pos.items(), key=lambda x: -x[1])),
        scores_neg=dict(sorted(scores_neg.items(), key=lambda x: -x[1])),
        precision=precision,
        K=K,
        steer_target=steer_target,
        n_calibration=n_calibration,
        pos_alphas=pos_alphas,
        neg_alphas=neg_alphas,
    )


def directional_rerank_report(
    scores_pos_context: dict[str, float],
    scores_pos_parametric: dict[str, float],
    *,
    K: int = 10,
) -> dict:
    """Offline-comparable directional re-rank on raw pos score maps."""
    keys = sorted(set(scores_pos_context) | set(scores_pos_parametric))
    rows = []
    for k in keys:
        c = float(scores_pos_context.get(k, 0.0))
        p = float(scores_pos_parametric.get(k, 0.0))
        rows.append(
            {
                "head": k,
                "ctx_pos": c,
                "par_pos": p,
                "directional_ctx": c - p,
                "directional_par": p - c,
            }
        )
    by_ctx = sorted(rows, key=lambda r: -r["directional_ctx"])
    by_par = sorted(rows, key=lambda r: -r["directional_par"])
    set_c = {r["head"] for r in by_ctx[:K]}
    set_p = {r["head"] for r in by_par[:K]}
    inter = set_c & set_p
    union = set_c | set_p
    return {
        "K": K,
        "directional_ctx_top": by_ctx[:K],
        "directional_par_top": by_par[:K],
        "jaccard": (len(inter) / len(union)) if union else 1.0,
        "intersection": sorted(inter),
    }


def _top_k_heads(scores: dict[str, float], K: int) -> list[HeadRef]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]
    out: list[HeadRef] = []
    for k, _ in ranked:
        l_str, h_str = k.replace("L", "").split("_H")
        out.append(HeadRef(int(l_str), int(h_str)))
    return out


def _dump_partial_scores(path, pos_shifts, neg_shifts, steer_target: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "steer_target": steer_target,
        "scores_pos_mean": {k: sum(v) / len(v) for k, v in pos_shifts.items() if v},
        "scores_neg_mean": {k: sum(v) / len(v) for k, v in neg_shifts.items() if v},
        "n_obs_pos": {k: len(v) for k, v in pos_shifts.items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _intervention_list(
    identified: IdentifiedHeads,
    alpha_plus: float,
    alpha_minus: float,
) -> list[tuple[int, int, float]]:
    """Build (layer, head, alpha) list: +α on H+, −α on H−."""
    out: list[tuple[int, int, float]] = []
    if alpha_plus != 0:
        for h in identified.enlarge:
            out.append((h.layer, h.head, alpha_plus))
    if alpha_minus != 0:
        for h in identified.suppress:
            out.append((h.layer, h.head, -abs(alpha_minus)))
    return out


@torch.inference_mode()
def greedy_generate_juice(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    interventions: Sequence[tuple[int, int, float]],
    *,
    max_new_tokens: int = 16,
) -> str:
    """
    Faithful JuICE generate_with_intervention:
    for each new token, pass 1 records head outs; pass 2 injects and argmaxes.
    """
    wrap_llama_attention_for_juice(model)
    dev = input_device(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    generated = input_ids.clone()
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        step_inputs = {"input_ids": generated, "attention_mask": torch.ones_like(generated)}

        # Pass 1: record head contributions for the *current* sequence.
        _set_head_out(model, True)
        _ = model(**step_inputs, use_cache=False)
        cached = _collect_all_head_outputs(model)
        _set_head_out(model, False)

        # Pass 2: inject cached outs, read logits.
        specs = [
            (ln, hn, alph, cached[(ln, hn)])
            for ln, hn, alph in interventions
            if (ln, hn) in cached
        ]
        handles = _inject_o_proj_hooks(model, specs)
        try:
            logits = _forward_logits(model, step_inputs)
        finally:
            for h in handles:
                h.remove()

        next_id = int(torch.argmax(logits).item())
        next_tok = torch.tensor([[next_id]], device=dev, dtype=generated.dtype)
        generated = torch.cat([generated, next_tok], dim=-1)
        del cached
        if eos_id is not None and next_id == eos_id:
            break

    gen_ids = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def conflict_eval_prompt(
    tokenizer: PreTrainedTokenizerBase,
    item: ConflictItem,
    *,
    prompt_format: Literal["instruct_chat", "juice_coherent"] = "instruct_chat",
) -> str:
    """Eval conflict prompt: Instruct chat template vs JuICE coherent-conflict completion."""
    if prompt_format == "instruct_chat":
        return apply_chat_template(tokenizer, item, "context_conflicts")
    if prompt_format == "juice_coherent":
        return juice_style_factual_prompts(item)["coherent_conflict"]
    raise ValueError(prompt_format)


@torch.inference_mode()
def apply_juice_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[ConflictItem],
    identified: IdentifiedHeads,
    *,
    alpha_plus: float = 5.0,
    alpha_minus: float = 2.0,
    alpha: float | None = None,
    max_new_tokens: int = 16,
    prompt_format: Literal["instruct_chat", "juice_coherent"] = "instruct_chat",
) -> list[dict]:
    # Legacy kwarg: single `alpha` => enlarge-only (no H−), matching old scripts.
    if alpha is not None:
        alpha_plus = alpha
        alpha_minus = 0.0
        if identified.suppress and alpha_minus == 0.0:
            # Keep suppress inactive when caller only passed `alpha`.
            pass
    interventions = _intervention_list(identified, alpha_plus, alpha_minus)
    results = []
    for item in tqdm(items, desc=f"juice_apply[{identified.steer_target}]"):
        prompt = conflict_eval_prompt(tokenizer, item, prompt_format=prompt_format)
        if interventions:
            text = greedy_generate_juice(
                model, tokenizer, prompt, interventions, max_new_tokens=max_new_tokens
            )
        else:
            from common.inference import greedy_generate

            text = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        label = label_conflict_response(text, item.parametric_answer, item.context_answer)
        results.append(
            {
                "id": item.id,
                "text": text,
                "label": label,
                "steer_target": identified.steer_target,
                "follows_context": label == "context",
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def random_head_control(
    n_layers: int,
    n_heads: int,
    K: int,
    *,
    seed: int,
    exclude: Iterable[HeadRef] | None = None,
) -> list[HeadRef]:
    exclude_set = {h.key() for h in (exclude or [])}
    pool = [
        HeadRef(l, h)
        for l in range(n_layers)
        for h in range(n_heads)
        if HeadRef(l, h).key() not in exclude_set
    ]
    rng = random.Random(seed)
    return rng.sample(pool, k=min(K, len(pool)))


def jaccard(a: Sequence[HeadRef], b: Sequence[HeadRef]) -> float:
    sa, sb = {h.key() for h in a}, {h.key() for h in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def context_follow_rate(results: list[dict]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.get("follows_context") or r["label"] == "context") / len(
        results
    )


def steering_efficacy(results: list[dict], target: SteerTarget) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r["label"] == target) / len(results)


def print_identified(ident: IdentifiedHeads, top_n: int = 10) -> None:
    print(f"\n=== Identified heads [{ident.steer_target}] K={ident.K} ===")
    print("Enlarge (H+):")
    for h in ident.enlarge:
        print(f"  {h.key():12s}  score_pos={ident.scores_pos.get(h.key(), float('nan')):.6f}")
    print("Suppress (H−):")
    for h in ident.suppress:
        print(f"  {h.key():12s}  score_neg={ident.scores_neg.get(h.key(), float('nan')):.6f}")
    print(f"Top-{top_n} positive scores:")
    for i, (k, s) in enumerate(list(ident.scores_pos.items())[:top_n]):
        print(f"  {i+1:2d}. {k:12s}  {s:.6f}")
