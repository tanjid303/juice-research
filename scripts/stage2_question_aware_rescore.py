"""Stage-2-wide question-aware relabel + paired bootstrap.

Does not rewrite original run JSONs. Writes:
  stage2_question_aware_rescore/relabeled/*.json  (audit trail)
  stage2_question_aware_rescore/headline_first.json
  stage2_question_aware_rescore/master_comparison.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\tt")
sys.path.insert(0, str(ROOT))
from common.scoring import label_conflict_response  # noqa: E402

N_BOOT = 1000
SEED = 42
DATA = ROOT / "data" / "conflictqa_pilot250_popular.jsonl"
OUT = ROOT / "stage2_question_aware_rescore"
REL = OUT / "relabeled"

P = ROOT  # shorthand


def p(*parts: str) -> str:
    return str(P.joinpath(*parts))


# Each cell: id, family, condition, n, baseline path, steered path, headline?
CELLS: list[dict] = [
    # --- Llama n=100 headline ---
    dict(id="llama_n100_fp16_ctx_k10", family="Llama n=100 FP16", condition="context K10 enlarge", n=100, headline=True,
         base=p("llama1", "juice_stage2_llama_base_n100", "steer_baseline.json"),
         steered=p("llama1", "juice_stage2_llama_base_n100", "steer_context_k10.json")),
    dict(id="llama_n100_fp16_par_k2", family="Llama n=100 FP16", condition="parametric clean K2", n=100, headline=True,
         base=p("llama1", "juice_stage2_llama_base_n100", "steer_baseline.json"),
         steered=p("llama1", "juice_stage2_llama_base_n100", "steer_parametric_k2_clean.json")),
    dict(id="llama_n100_fp16_rand_k10", family="Llama n=100 FP16", condition="random context K10", n=100, headline=True,
         base=p("llama1", "juice_stage2_llama_base_n100", "steer_baseline.json"),
         steered=p("llama1", "juice_stage2_llama_base_n100", "random_context_k10.json")),
    dict(id="llama_n100_int4_ctx_k10", family="Llama n=100 INT4 transfer", condition="context K10 enlarge", n=100, headline=True,
         base=p("llama2", "juice_stage2_int4_ncalib128_n100", "steer_baseline.json"),
         steered=p("llama2", "juice_stage2_int4_ncalib128_n100", "steer_context_k10.json")),
    dict(id="llama_n100_int4_par_k2", family="Llama n=100 INT4 transfer", condition="parametric clean K2", n=100, headline=True,
         base=p("llama2", "juice_stage2_int4_ncalib128_n100", "steer_baseline.json"),
         steered=p("llama2", "juice_stage2_int4_ncalib128_n100", "steer_parametric_k2_clean.json")),
    dict(id="llama_n100_int4_rand_k10", family="Llama n=100 INT4 transfer", condition="random context K10", n=100, headline=True,
         base=p("llama2", "juice_stage2_int4_ncalib128_n100", "steer_baseline.json"),
         steered=p("llama2", "juice_stage2_int4_ncalib128_n100", "random_context_k10.json")),
    dict(id="llama_n100_int4_native_k10", family="Llama n=100 INT4 native", condition="native context K10", n=100, headline=True,
         base=p("hobe-kicu", "llama3_int4_n100_remaining", "steer_baseline.json"),
         steered=p("hobe-kicu", "llama3_int4_n100_remaining", "native_context_k10.json")),
    dict(id="llama_n100_int4_hminus", family="Llama n=100 INT4 native", condition="context H− K10", n=100, headline=False,
         base=p("hobe-kicu", "llama3_int4_n100_remaining", "steer_baseline.json"),
         steered=p("hobe-kicu", "llama3_int4_n100_remaining", "context_hminus_k10.json")),
    # --- Gemma n=50 (abstract) ---
    dict(id="gemma_n50_fp16_ctx_k10", family="Gemma n=50 FP16", condition="context K10 enlarge", n=50, headline=True,
         base=p("juice_stage2_gemma2_base_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_gemma2_base_cell3", "steer_context_k10.json")),
    dict(id="gemma_n50_fp16_l27", family="Gemma n=50 FP16", condition="raw L27_H6", n=50, headline=True,
         base=p("juice_stage2_gemma2_base_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_gemma2_base_cell3", "steer_parametric_k1_raw.json")),
    dict(id="gemma_n50_fp16_par_k2", family="Gemma n=50 FP16", condition="parametric clean K2", n=50, headline=True,
         base=p("juice_stage2_gemma2_base_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_gemma2_base_cell3", "steer_parametric_k2_clean.json")),
    dict(id="gemma_n50_fp16_rand_k10", family="Gemma n=50 FP16", condition="random context K10", n=50, headline=False,
         base=p("juice_stage2_gemma2_base_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_gemma2_base_cell3", "random_context_k10.json")),
    dict(id="gemma_n50_int4_l27", family="Gemma n=50 INT4", condition="raw L27_H6", n=50, headline=True,
         base=p("result-task-1", "juice_stage2_gemma2_int4_transfer_l27h6", "steer_baseline.json"),
         steered=p("result-task-1", "juice_stage2_gemma2_int4_transfer_l27h6", "steer_parametric_k1_raw.json")),
    dict(id="gemma_n50_int4_rand_k1", family="Gemma n=50 INT4", condition="random K1", n=50, headline=True,
         base=p("result-task-1", "juice_stage2_gemma2_int4_transfer_l27h6", "steer_baseline.json"),
         steered=p("result-task-1", "juice_stage2_gemma2_int4_transfer_l27h6", "random_parametric_k1.json")),
    # --- Gemma n=100 ---
    dict(id="gemma_n100_fp16_ctx_k10", family="Gemma n=100 FP16", condition="context K10 enlarge", n=100, headline=True,
         base=p("last-one-g", "gemma_n100_fp16", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_fp16", "steer_context_k10.json")),
    dict(id="gemma_n100_fp16_l27", family="Gemma n=100 FP16", condition="raw L27_H6", n=100, headline=True,
         base=p("last-one-g", "gemma_n100_fp16", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_fp16", "steer_parametric_k1_raw.json")),
    dict(id="gemma_n100_fp16_par_k2", family="Gemma n=100 FP16", condition="parametric clean K2", n=100, headline=True,
         base=p("last-one-g", "gemma_n100_fp16", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_fp16", "steer_parametric_k2_clean.json")),
    dict(id="gemma_n100_fp16_rand_k10", family="Gemma n=100 FP16", condition="random context K10", n=100, headline=False,
         base=p("last-one-g", "gemma_n100_fp16", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_fp16", "random_context_k10.json")),
    dict(id="gemma_n100_int4_ctx_k10", family="Gemma n=100 INT4", condition="context K10 enlarge", n=100, headline=True,
         base=p("last-one-g", "gemma_n100_int4_headline", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_int4_headline", "steer_context_k10.json")),
    dict(id="gemma_n100_int4_l27", family="Gemma n=100 INT4", condition="raw L27_H6", n=100, headline=True,
         base=p("last-one-g", "gemma_n100_int4_headline", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_int4_headline", "steer_parametric_k1_raw.json")),
    dict(id="gemma_n100_int4_rand_k10", family="Gemma n=100 INT4", condition="random context K10", n=100, headline=False,
         base=p("last-one-g", "gemma_n100_int4_headline", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_int4_headline", "random_context_k10.json")),
    dict(id="gemma_n100_int4_rand_k1", family="Gemma n=100 INT4", condition="random K1", n=100, headline=True,
         base=p("last-one-g", "gemma_n100_int4_headline", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_int4_headline", "random_parametric_k1.json")),
    # --- Llama n=50 FP16 ---
    dict(id="llama_n50_fp16_ctx_k10", family="Llama n=50 FP16", condition="context K10 enlarge", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_base_directional_cell3", "steer_context_k10.json")),
    dict(id="llama_n50_fp16_ctx_k5", family="Llama n=50 FP16", condition="context K5 enlarge", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_base_directional_cell3", "steer_context_k5.json")),
    dict(id="llama_n50_fp16_par_k2", family="Llama n=50 FP16", condition="parametric K2", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_base_directional_cell3", "steer_parametric_k2.json")),
    dict(id="llama_n50_fp16_par_k2c", family="Llama n=50 FP16", condition="parametric clean K2", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_base_directional_cell3", "steer_parametric_k2_clean.json")),
    dict(id="llama_n50_fp16_par_k3", family="Llama n=50 FP16", condition="parametric K3", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("juice_stage2_base_directional_cell3", "steer_parametric_k3.json")),
    # --- Llama n=50 INT4 N64 ---
    dict(id="llama_n50_int4_n64_ctx_k10", family="Llama n=50 INT4 N64", condition="context K10 enlarge", n=50,
         base=p("int_4_result", "steer_baseline (1).json"),
         steered=p("int_4_result", "steer_context_k10 (1).json")),
    dict(id="llama_n50_int4_n64_ctx_k5", family="Llama n=50 INT4 N64", condition="context K5 enlarge", n=50,
         base=p("int_4_result", "steer_baseline (1).json"),
         steered=p("int_4_result", "steer_context_k5 (1).json")),
    dict(id="llama_n50_int4_n64_par_k2", family="Llama n=50 INT4 N64", condition="parametric K2", n=50,
         base=p("int_4_result", "steer_baseline (1).json"),
         steered=p("int_4_result", "steer_parametric_k2 (1).json")),
    dict(id="llama_n50_int4_n64_par_k2c", family="Llama n=50 INT4 N64", condition="parametric clean K2", n=50,
         base=p("int_4_result", "steer_baseline (1).json"),
         steered=p("int_4_result", "steer_parametric_k2_clean.json")),
    dict(id="llama_n50_int4_n64_par_k3", family="Llama n=50 INT4 N64", condition="parametric K3", n=50,
         base=p("int_4_result", "steer_baseline (1).json"),
         steered=p("int_4_result", "steer_parametric_k3 (1).json")),
    # --- Llama n=50 INT4 random (N64 session) ---
    dict(id="llama_n50_int4_rand_ctx_k10", family="Llama n=50 INT4 N64", condition="random context K10", n=50,
         base=p("here_is_the", "steer_baseline (1).json"),
         steered=p("here_is_the", "random_context_k10.json")),
    dict(id="llama_n50_int4_rand_ctx_k5", family="Llama n=50 INT4 N64", condition="random context K5", n=50,
         base=p("here_is_the", "steer_baseline (1).json"),
         steered=p("here_is_the", "random_context_k5.json")),
    dict(id="llama_n50_int4_rand_par_k2", family="Llama n=50 INT4 N64", condition="random parametric K2", n=50,
         base=p("here_is_the", "steer_baseline (1).json"),
         steered=p("here_is_the", "random_parametric_k2.json")),
    dict(id="llama_n50_int4_rand_par_k3", family="Llama n=50 INT4 N64", condition="random parametric K3", n=50,
         base=p("here_is_the", "steer_baseline (1).json"),
         steered=p("here_is_the", "random_parametric_k3.json")),
    # --- Llama n=50 INT4 N128 ---
    dict(id="llama_n50_int4_n128_ctx_k10", family="Llama n=50 INT4 N128", condition="context K10 enlarge", n=50,
         base=p("this_is_the_128", "steer_baseline (2).json"),
         steered=p("this_is_the_128", "steer_context_k10 (1).json")),
    dict(id="llama_n50_int4_n128_par_k2c", family="Llama n=50 INT4 N128", condition="parametric clean K2", n=50,
         base=p("this_is_the_128", "steer_baseline (2).json"),
         steered=p("this_is_the_128", "steer_parametric_k2_clean.json")),
    dict(id="llama_n50_int4_n128_rand_k10", family="Llama n=50 INT4 N128", condition="random context K10", n=50,
         base=p("this_is_the_128", "steer_baseline (2).json"),
         steered=p("this_is_the_128", "random_context_k10.json")),
    # --- Llama n=50 repairs ---
    dict(id="llama_n50_int4_l1h4", family="Llama n=50 INT4 repairs", condition="raw peak L1_H4", n=50,
         base=p("here_is_the-2", "steer_baseline (1).json"),
         steered=p("here_is_the-2", "steer_l1_h4.json")),
    dict(id="llama_n50_int4_hminus", family="Llama n=50 INT4 repairs", condition="context H− K10", n=50,
         base=p("here_is_the-2", "steer_baseline (1).json"),
         steered=p("here_is_the-2", "steer_suppress_ctx_k10.json")),
    dict(id="llama_n50_int4_a05", family="Llama n=50 INT4 repairs", condition="clean K2 α=0.5", n=50,
         base=p("here_is_the-2", "steer_baseline (1).json"),
         steered=p("here_is_the-2", "steer_clean_alpha_0.5.json")),
    dict(id="llama_n50_int4_a1", family="Llama n=50 INT4 repairs", condition="clean K2 α=1.0", n=50,
         base=p("here_is_the-2", "steer_baseline (1).json"),
         steered=p("here_is_the-2", "steer_clean_alpha_1.0.json")),
    dict(id="llama_n50_int4_a2", family="Llama n=50 INT4 repairs", condition="clean K2 α=2.0", n=50,
         base=p("here_is_the-2", "steer_baseline (1).json"),
         steered=p("here_is_the-2", "steer_clean_alpha_2.0.json")),
    dict(id="llama_n50_int4_a3", family="Llama n=50 INT4 repairs", condition="clean K2 α=3.0", n=50,
         base=p("here_is_the-2", "steer_baseline (1).json"),
         steered=p("here_is_the-2", "steer_clean_alpha_3.0.json")),
    dict(id="llama_n50_int4_a5", family="Llama n=50 INT4 repairs", condition="clean K2 α=5.0", n=50,
         base=p("here_is_the-2", "steer_baseline (1).json"),
         steered=p("here_is_the-2", "steer_clean_alpha_5.0.json")),
    # --- Llama n=50 native / reverse ---
    dict(id="llama_n50_native_par_k2c", family="Llama n=50 INT4 native", condition="native parametric clean K2", n=50,
         base=p("ungabunga_result", "juice_stage2_int4_ncalib128", "steer_baseline_native_clean_k2_cell.json"),
         steered=p("ungabunga_result", "juice_stage2_int4_ncalib128", "steer_int4_native_parametric_k2_clean.json")),
    dict(id="llama_n50_native_ctx_k10", family="Llama n=50 INT4 native", condition="native context K10", n=50,
         base=p("this_is_the_128", "steer_baseline (2).json"),
         steered=p("unga-bunga-23", "juice_stage2_int4_ncalib128", "steer_native_int4_context_k10.json")),
    dict(id="llama_n50_reverse_a1", family="Llama n=50 reverse", condition="INT4 K10 on FP16 α=1", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("unga-bunga-23", "juice_stage2_int4_ncalib128", "steer_reverse_int4_context_k10_on_fp16.json")),
    dict(id="llama_n50_reverse_a05", family="Llama n=50 reverse", condition="INT4 K10 on FP16 α=0.5", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("resultofthe", "juice_stage2_int4_ncalib128", "steer_reverse_int4_k10_fp16_a0.5.json")),
    dict(id="llama_n50_reverse_a025", family="Llama n=50 reverse", condition="INT4 K10 on FP16 α=0.25", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("resultofthe", "juice_stage2_int4_ncalib128", "steer_reverse_int4_k10_fp16_a0.25.json")),
    dict(id="llama_n50_reverse_a1_dup", family="Llama n=50 reverse", condition="INT4 K10 on FP16 α=1 (resultofthe)", n=50,
         base=p("juice_stage2_base_directional_cell3", "steer_baseline.json"),
         steered=p("resultofthe", "juice_stage2_int4_ncalib128", "steer_reverse_int4_k10_fp16_a1.0.json")),
    # --- Llama n=50 FP16 alpha sweep ---
    dict(id="llama_n50_fp16_ctx_a1", family="Llama n=50 FP16 α-sweep", condition="context α+=1 α−=0", n=50,
         base=p("juice_stage2_alpha_sweep", "steer_baseline.json"),
         steered=p("juice_stage2_alpha_sweep", "steer_context_ap1p0_am0p0.json")),
    dict(id="llama_n50_fp16_ctx_a2", family="Llama n=50 FP16 α-sweep", condition="context α+=2 α−=1", n=50,
         base=p("juice_stage2_alpha_sweep", "steer_baseline.json"),
         steered=p("juice_stage2_alpha_sweep", "steer_context_ap2p0_am1p0.json")),
    dict(id="llama_n50_fp16_par_a1", family="Llama n=50 FP16 α-sweep", condition="parametric α+=1 α−=0", n=50,
         base=p("juice_stage2_alpha_sweep", "steer_baseline.json"),
         steered=p("juice_stage2_alpha_sweep", "steer_parametric_ap1p0_am0p0.json")),
    dict(id="llama_n50_fp16_par_a2", family="Llama n=50 FP16 α-sweep", condition="parametric α+=2 α−=1", n=50,
         base=p("juice_stage2_alpha_sweep", "steer_baseline.json"),
         steered=p("juice_stage2_alpha_sweep", "steer_parametric_ap2p0_am1p0.json")),
    # --- Llama n=100 INT4 diagnostics ---
    dict(id="llama_n100_diag_noisy_k2", family="Llama n=100 INT4 diagnostics", condition="noisy parametric K2", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "noisy_parametric_k2.json")),
    dict(id="llama_n100_diag_noisy_k3", family="Llama n=100 INT4 diagnostics", condition="noisy parametric K3", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "noisy_parametric_k3.json")),
    dict(id="llama_n100_diag_l1h4", family="Llama n=100 INT4 diagnostics", condition="raw peak L1_H4", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "raw_peak_l1_h4.json")),
    dict(id="llama_n100_diag_native_k2", family="Llama n=100 INT4 diagnostics", condition="native parametric K2", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "native_parametric_k2.json")),
    dict(id="llama_n100_diag_a05", family="Llama n=100 INT4 diagnostics", condition="clean K2 α=0.5", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "clean_parametric_alpha_0.5.json")),
    dict(id="llama_n100_diag_a1", family="Llama n=100 INT4 diagnostics", condition="clean K2 α=1", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "clean_parametric_alpha_1.json")),
    dict(id="llama_n100_diag_a2", family="Llama n=100 INT4 diagnostics", condition="clean K2 α=2", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "clean_parametric_alpha_2.json")),
    dict(id="llama_n100_diag_a3", family="Llama n=100 INT4 diagnostics", condition="clean K2 α=3", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "clean_parametric_alpha_3.json")),
    dict(id="llama_n100_diag_a5", family="Llama n=100 INT4 diagnostics", condition="clean K2 α=5", n=100,
         base=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "steer_baseline.json"),
         steered=p("lama-last-maybe", "llama3_int4_n100_diagnostics", "clean_parametric_alpha_5.json")),
    # --- Gemma n=50 H− ---
    dict(id="gemma_n50_fp16_hminus_id", family="Gemma n=50 FP16 H−", condition="identified H−", n=50,
         base=p("task-2-res", "juice_stage2_gemma2_fp16_context_suppress", "steer_baseline.json"),
         steered=p("task-2-res", "juice_stage2_gemma2_fp16_context_suppress", "suppress_context_k10.json")),
    dict(id="gemma_n50_fp16_hminus_rand", family="Gemma n=50 FP16 H−", condition="random H−", n=50,
         base=p("res-fp16-test-gemma", "juice_stage2_gemma2_fp16_random_k10_suppress", "steer_baseline.json"),
         steered=p("res-fp16-test-gemma", "juice_stage2_gemma2_fp16_random_k10_suppress", "random_suppress_k10.json")),
    dict(id="gemma_n50_int4_hminus_id", family="Gemma n=50 INT4 H−", condition="identified H−", n=50,
         base=p("gemma-int4-kicu", "juice_stage2_gemma2_int4_context_suppress", "steer_baseline.json"),
         steered=p("gemma-int4-kicu", "juice_stage2_gemma2_int4_context_suppress", "suppress_context_k10.json")),
    dict(id="gemma_n50_int4_hminus_rand", family="Gemma n=50 INT4 H−", condition="random H−", n=50,
         base=p("gemma-int4-kicu", "juice_stage2_gemma2_int4_context_suppress", "steer_baseline.json"),
         steered=p("gemma-int4-kicu", "juice_stage2_gemma2_int4_context_suppress", "random_suppress_k10.json")),
    # --- Gemma n=100 H− ---
    dict(id="gemma_n100_fp16_hminus_id", family="Gemma n=100 FP16 H−", condition="identified H−", n=100,
         base=p("last-one-g", "gemma_n100_fp16_hminus", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_fp16_hminus", "suppress_context_k10.json")),
    dict(id="gemma_n100_fp16_hminus_rand", family="Gemma n=100 FP16 H−", condition="random H−", n=100,
         base=p("last-one-g", "gemma_n100_fp16_random_hminus", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_fp16_random_hminus", "random_suppress_k10.json")),
    dict(id="gemma_n100_int4_hminus_id", family="Gemma n=100 INT4 H−", condition="identified H−", n=100,
         base=p("last-one-g", "gemma_n100_int4_hminus", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_int4_hminus", "suppress_context_k10.json")),
    dict(id="gemma_n100_int4_hminus_rand", family="Gemma n=100 INT4 H−", condition="random H−", n=100,
         base=p("last-one-g", "gemma_n100_int4_hminus", "steer_baseline.json"),
         steered=p("last-one-g", "gemma_n100_int4_hminus", "random_suppress_k10.json")),
]


def load_gold() -> dict:
    out = {}
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            out[o["id"]] = o
    return gold_ok(out)


def gold_ok(out: dict) -> dict:
    return out


def load_rows(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    raise ValueError(f"not a row list: {path}")


def relabel_file(path: Path, gold: dict) -> list[dict]:
    rows = load_rows(path)
    out = []
    for r in rows:
        iid = r["id"]
        text = r.get("text") or r.get("completion") or ""
        g = gold[iid]
        old = r.get("label") or r.get("conflict_label")
        new = label_conflict_response(
            text, g["parametric_answer"], g["context_answer"], question=g["question"]
        )
        out.append(
            {
                "id": iid,
                "text": text,
                "original_label": old,
                "rescored_label": new,
                "changed": old != new,
                "original_follows_context": bool(
                    r.get("follows_context") or old == "context"
                ),
                "rescored_follows_context": new == "context",
                "steer_target": r.get("steer_target"),
            }
        )
    return out


def pco(labels: list[str]) -> dict:
    c = Counter(labels)
    return {
        "parametric": int(c.get("parametric", 0)),
        "context": int(c.get("context", 0)),
        "rejects_context": int(c.get("rejects_context", 0)),
        "other": int(c.get("other", 0)),
        "pco": f"{c.get('parametric', 0)}/{c.get('context', 0)}/{c.get('other', 0)}",
    }


def fmt_pp(x: float) -> str:
    v = round(x)
    if abs(x - v) < 1e-6:
        return f"{v:+.0f}"
    return f"{x:+.1f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt_pp(lo)}, {fmt_pp(hi)}]"


def call_of(excl: bool) -> str:
    return "detected" if excl else "null"


def sign_of(x: float) -> str:
    if x > 1e-12:
        return "+"
    if x < -1e-12:
        return "−"
    return "0"


def bootstrap_paired_follow_delta(
    baseline_rows: list[dict],
    steered_rows: list[dict],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    base_by = {
        r["id"]: bool(r.get("follows_context") or r.get("label") == "context")
        for r in baseline_rows
    }
    ste_by = {
        r["id"]: bool(r.get("follows_context") or r.get("label") == "context")
        for r in steered_rows
    }
    ids = sorted(set(base_by) & set(ste_by))
    n = len(ids)

    def rate(id_list: list[str], table: dict[str, bool]) -> float:
        return sum(table[i] for i in id_list) / len(id_list)

    r_base = rate(ids, base_by)
    r_ste = rate(ids, ste_by)
    d_point = r_ste - r_base
    rng = np.random.default_rng(seed)
    deltas, bases, stes = [], [], []
    for _ in range(n_boot):
        draw = [ids[i] for i in rng.integers(0, n, size=n)]
        rb = rate(draw, base_by)
        rs = rate(draw, ste_by)
        bases.append(rb)
        stes.append(rs)
        deltas.append(rs - rb)

    def pct_ci(xs: list[float]) -> list[float]:
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    d_lo, d_hi = pct_ci(deltas)
    return {
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "baseline_fc": {"point": r_base, "ci95": pct_ci(bases)},
        "steered_fc": {"point": r_ste, "ci95": pct_ci(stes)},
        "delta": {
            "point": d_point,
            "ci95": [d_lo, d_hi],
            "point_pp": 100.0 * d_point,
            "ci95_pp": [100.0 * d_lo, 100.0 * d_hi],
            "excludes_zero": not (d_lo <= 0.0 <= d_hi),
            "n_draws_lt0": sum(1 for x in deltas if x < 0),
            "n_draws_eq0": sum(1 for x in deltas if x == 0),
            "n_draws_gt0": sum(1 for x in deltas if x > 0),
        },
    }


def boot_with_draws(base_fc: dict[str, bool], ste_fc: dict[str, bool]) -> dict:
    ids = sorted(set(base_fc) & set(ste_fc))
    n = len(ids)
    rng = np.random.default_rng(SEED)
    point = sum(ste_fc[i] for i in ids) / n - sum(base_fc[i] for i in ids) / n
    draws = []
    for _ in range(N_BOOT):
        draw = [ids[j] for j in rng.integers(0, n, size=n)]
        draws.append(
            sum(ste_fc[i] for i in draw) / n - sum(base_fc[i] for i in draw) / n
        )
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {
        "n": n,
        "delta_pp": 100 * point,
        "ci95_pp": [100 * lo, 100 * hi],
        "excludes_zero": not (lo <= 0.0 <= hi),
        "n_draws_lt0": sum(x < 0 for x in draws),
        "n_draws_eq0": sum(x == 0 for x in draws),
        "n_draws_gt0": sum(x > 0 for x in draws),
        "draws": draws,
        "point": point,
        "fc_base": sum(base_fc[i] for i in ids) / n,
        "fc_ste": sum(ste_fc[i] for i in ids) / n,
    }


def as_boot_rows(relabeled: list[dict], which: str) -> list[dict]:
    key_lab = "original_label" if which == "original" else "rescored_label"
    key_fc = "original_follows_context" if which == "original" else "rescored_follows_context"
    return [
        {"id": r["id"], "label": r[key_lab], "follows_context": r[key_fc]}
        for r in relabeled
    ]


def analyze_cell(cell: dict, cache: dict[str, list[dict]]) -> dict:
    base = cache[str(Path(cell["base"]).resolve())]
    ste = cache[str(Path(cell["steered"]).resolve())]
    base_by = {r["id"]: r for r in base}
    ste_by = {r["id"]: r for r in ste}
    ids = sorted(set(base_by) & set(ste_by))
    n = len(ids)
    if n != cell["n"]:
        raise ValueError(f"{cell['id']}: expected n={cell['n']}, got {n}")

    orig_boot = bootstrap_paired_follow_delta(
        as_boot_rows(base, "original"), as_boot_rows(ste, "original"), n_boot=N_BOOT, seed=SEED
    )
    new_boot = bootstrap_paired_follow_delta(
        as_boot_rows(base, "rescored"), as_boot_rows(ste, "rescored"), n_boot=N_BOOT, seed=SEED
    )
    orig_d = orig_boot["delta"]
    new_d = new_boot["delta"]
    orig_call = call_of(orig_d["excludes_zero"])
    new_call = call_of(new_d["excludes_zero"])
    orig_sign = sign_of(orig_d["point_pp"])
    new_sign = sign_of(new_d["point_pp"])
    call_unchanged = orig_call == new_call
    sign_changed = orig_sign != new_sign
    flagged = (not call_unchanged) or sign_changed

    n_chg_ste = sum(1 for i in ids if ste_by[i]["changed"])
    n_chg_base = sum(1 for i in ids if base_by[i]["changed"])

    orig_ste_labs = [ste_by[i]["original_label"] for i in ids]
    new_ste_labs = [ste_by[i]["rescored_label"] for i in ids]
    orig_base_labs = [base_by[i]["original_label"] for i in ids]
    new_base_labs = [base_by[i]["rescored_label"] for i in ids]

    return {
        "id": cell["id"],
        "family": cell["family"],
        "condition": cell["condition"],
        "n": n,
        "headline": bool(cell.get("headline")),
        "baseline_path": cell["base"],
        "steered_path": cell["steered"],
        "original": {
            "fc_base": orig_boot["baseline_fc"]["point"],
            "fc_steered": orig_boot["steered_fc"]["point"],
            "delta_pp": orig_d["point_pp"],
            "ci95_pp": orig_d["ci95_pp"],
            "excludes_zero": orig_d["excludes_zero"],
            "call": orig_call,
            "sign": orig_sign,
            "fmt": f"{fmt_pp(orig_d['point_pp'])} {fmt_ci(*orig_d['ci95_pp'])}",
            "pco_steered": pco(orig_ste_labs),
            "pco_baseline": pco(orig_base_labs),
            "n_draws_lt0": orig_d["n_draws_lt0"],
            "n_draws_eq0": orig_d["n_draws_eq0"],
            "n_draws_gt0": orig_d["n_draws_gt0"],
        },
        "rescored": {
            "fc_base": new_boot["baseline_fc"]["point"],
            "fc_steered": new_boot["steered_fc"]["point"],
            "delta_pp": new_d["point_pp"],
            "ci95_pp": new_d["ci95_pp"],
            "excludes_zero": new_d["excludes_zero"],
            "call": new_call,
            "sign": new_sign,
            "fmt": f"{fmt_pp(new_d['point_pp'])} {fmt_ci(*new_d['ci95_pp'])}",
            "pco_steered": pco(new_ste_labs),
            "pco_baseline": pco(new_base_labs),
            "n_draws_lt0": new_d["n_draws_lt0"],
            "n_draws_eq0": new_d["n_draws_eq0"],
            "n_draws_gt0": new_d["n_draws_gt0"],
        },
        "n_labels_changed_steered": n_chg_ste,
        "n_labels_changed_baseline": n_chg_base,
        "call_unchanged": call_unchanged,
        "sign_changed": sign_changed,
        "flagged": flagged,
        "flag": (
            f"{orig_call}→{new_call}"
            + ("; sign change" if sign_changed else "")
        ),
    }


def pack_derived(name: str, family: str, orig: dict, new: dict) -> dict:
    orig_call = call_of(orig["excludes_zero"])
    new_call = call_of(new["excludes_zero"])
    orig_sign = sign_of(orig["delta_pp"])
    new_sign = sign_of(new["delta_pp"])
    return {
        "id": name,
        "family": family,
        "condition": name.split(":")[-1] if ":" in name else name,
        "n": orig.get("n"),
        "headline": False,
        "derived": True,
        "original": {
            **{k: orig[k] for k in ("delta_pp", "ci95_pp", "excludes_zero")},
            "call": orig_call,
            "sign": orig_sign,
            "fmt": f"{fmt_pp(orig['delta_pp'])} {fmt_ci(*orig['ci95_pp'])}",
        },
        "rescored": {
            **{k: new[k] for k in ("delta_pp", "ci95_pp", "excludes_zero")},
            "call": new_call,
            "sign": new_sign,
            "fmt": f"{fmt_pp(new['delta_pp'])} {fmt_ci(*new['ci95_pp'])}",
        },
        "call_unchanged": orig_call == new_call,
        "sign_changed": orig_sign != new_sign,
        "flagged": orig_call != new_call or orig_sign != new_sign,
        "flag": f"{orig_call}→{new_call}" + ("; sign change" if orig_sign != new_sign else ""),
    }


def derived_spec(cache: dict, gold: dict, cells_by_id: dict) -> list[dict]:
    """Identified-minus-random and DiD for Gemma H−, both n=50 and n=100."""
    out = []
    groups = [
        (
            "Gemma n=100 H− derived",
            "gemma_n100_fp16_hminus_id",
            "gemma_n100_fp16_hminus_rand",
            "gemma_n100_int4_hminus_id",
            "gemma_n100_int4_hminus_rand",
        ),
        (
            "Gemma n=50 H− derived",
            "gemma_n50_fp16_hminus_id",
            "gemma_n50_fp16_hminus_rand",
            "gemma_n50_int4_hminus_id",
            "gemma_n50_int4_hminus_rand",
        ),
    ]
    cell_lookup = {c["id"]: c for c in CELLS}
    for family, id_16, rand_16, id_4, rand_4 in groups:
        specs = {}
        for tag, cid in (("fp16_id", id_16), ("fp16_rand", rand_16), ("int4_id", id_4), ("int4_rand", rand_4)):
            spec_cell = cell_lookup[cid]
            base = cache[str(Path(spec_cell["base"]).resolve())]
            ste = cache[str(Path(spec_cell["steered"]).resolve())]
            ids = sorted({r["id"] for r in base} & {r["id"] for r in ste})
            base_by = {r["id"]: r for r in base}
            ste_by = {r["id"]: r for r in ste}
            for which in ("original", "rescored"):
                fc_b = {i: (base_by[i]["original_follows_context"] if which == "original" else base_by[i]["rescored_follows_context"]) for i in ids}
                fc_s = {i: (ste_by[i]["original_follows_context"] if which == "original" else ste_by[i]["rescored_follows_context"]) for i in ids}
                specs.setdefault(which, {})[tag] = boot_with_draws(fc_b, fc_s)

        def pack_contrast(a, b):
            point = a["point"] - b["point"]
            xs = [x - y for x, y in zip(a["draws"], b["draws"])]
            lo, hi = float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))
            return {
                "n": a["n"],
                "delta_pp": 100 * point,
                "ci95_pp": [100 * lo, 100 * hi],
                "excludes_zero": not (lo <= 0.0 <= hi),
            }

        for which, blob in specs.items():
            specs[which]["spec16"] = pack_contrast(blob["fp16_id"], blob["fp16_rand"])
            specs[which]["spec4"] = pack_contrast(blob["int4_id"], blob["int4_rand"])
            specs[which]["did"] = pack_contrast(
                {"point": blob["int4_id"]["point"] - blob["int4_rand"]["point"],
                 "draws": [x - y for x, y in zip(blob["int4_id"]["draws"], blob["int4_rand"]["draws"])],
                 "n": blob["int4_id"]["n"]},
                {"point": blob["fp16_id"]["point"] - blob["fp16_rand"]["point"],
                 "draws": [x - y for x, y in zip(blob["fp16_id"]["draws"], blob["fp16_rand"]["draws"])],
                 "n": blob["fp16_id"]["n"]},
            )
        out.append(pack_derived("specificity FP16 (id − random)", family, specs["original"]["spec16"], specs["rescored"]["spec16"]))
        out.append(pack_derived("specificity INT4 (id − random)", family, specs["original"]["spec4"], specs["rescored"]["spec4"]))
        out.append(pack_derived("DiD (spec INT4 − spec FP16)", family, specs["original"]["did"], specs["rescored"]["did"]))
    return out


def print_table(rows: list[dict], title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'family':<28} {'condition':<32} {'orig Δ [CI]':>22} {'rescored Δ [CI]':>22}  {'call':<18} flag")
    for r in rows:
        star = ""
        if r["flagged"]:
            star = "  ***"
        print(
            f"{r['family']:<28} {r['condition']:<32} {r['original']['fmt']:>22} "
            f"{r['rescored']['fmt']:>22}  {r['flag']:<18}{star}"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    gold = load_gold()
    OUT.mkdir(parents=True, exist_ok=True)
    REL.mkdir(parents=True, exist_ok=True)

    paths = []
    for c in CELLS:
        paths.append(Path(c["base"]))
        paths.append(Path(c["steered"]))
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)

    cache: dict[str, list[dict]] = {}
    print(f"Relabeling {len(unique)} unique completion files...")
    for i, path in enumerate(unique, 1):
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = relabel_file(path, gold)
        cache[str(path.resolve())] = rel
        safe = "__".join(path.parts[-3:]).replace(" ", "_")
        (REL / f"{safe}.json").write_text(json.dumps(relabeled_payload(path, rel), indent=2), encoding="utf-8")
        nchg = sum(1 for r in rel if r["changed"])
        print(f"  [{i:02d}/{len(unique)}] {nchg:3d}/{len(rel)} labels changed  {path}")

    results: list[dict] = []
    headline_rows: list[dict] = []
    errors = []
    for c in CELLS:
        try:
            row = analyze_cell(c, cache)
        except Exception as e:
            errors.append({"id": c["id"], "error": repr(e)})
            print("ERROR", c["id"], e)
            continue
        results.append(row)
        if row["headline"]:
            headline_rows.append(row)

    print_table(headline_rows, "HEADLINE FIRST (load-bearing abstract / Llama+Gemma)")
    (OUT / "headline_first.json").write_text(
        json.dumps(
            {
                "matcher": "label_conflict_response(..., question=gold.question)",
                "n_boot": N_BOOT,
                "seed": SEED,
                "n_cells": len(headline_rows),
                "cells": headline_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote", OUT / "headline_first.json")

    derived = derived_spec(cache, gold, {r["id"]: r for r in results})
    results.extend(derived)
    print_table(results, "ALL CELLS")
    flagged = [r for r in results if r["flagged"]]
    print()
    print(f"Flagged cells (call flip and/or sign change): {len(flagged)} / {len(results)}")
    for r in flagged:
        print(f"  * {r['family']} | {r['condition']}: {r['original']['fmt']} → {r['rescored']['fmt']}  {r['flag']}")

    payload = {
        "matcher": "label_conflict_response(..., question=gold.question)",
        "protocol": "paired bootstrap n_boot=1000 seed=42; original labels vs question-aware labels; original run JSONs not rewritten",
        "n_unique_files": len(unique),
        "n_cells": len(results),
        "n_flagged": len(flagged),
        "errors": errors,
        "cells": results,
    }
    (OUT / "master_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT / "master_comparison.json")
    if errors:
        print("ERRORS", errors)


def relabeled_payload(path: Path, rel: list[dict]) -> dict:
    return {
        "source": str(path),
        "n": len(rel),
        "n_changed": sum(1 for r in rel if r["changed"]),
        "matcher": "label_conflict_response(..., question=gold.question)",
        "rows": rel,
    }


if __name__ == "__main__":
    main()
