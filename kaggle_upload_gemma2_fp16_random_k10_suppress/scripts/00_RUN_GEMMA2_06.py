"""Kaggle: run FP16 random-K10 suppress. Paste this as the only cell.

Do not upload _make_*.py. No INT4 checkpoint.
"""
from pathlib import Path

print("THIS IS GEMMA2_06 — FP16 random-K10 suppress. Not a notebook generator.")
hits = list(Path("/kaggle/input").rglob("gemma2_06_fp16_random_k10_suppress.py"))
hits += list(Path("/kaggle/working").rglob("gemma2_06_fp16_random_k10_suppress.py"))
hits = [h for h in hits if h.is_file()]
assert hits, "Add Data: kaggle_juice_gemma2_fp16_random_k10_suppress.zip"
script = hits[0]
print("SCRIPT", script)
exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"))
