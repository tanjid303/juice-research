"""Kaggle: run INT4 context-K10 suppress transfer. Paste this as the only cell.

Do not upload _make_*.py. Attach this zip + gemma-quantize N_CALIB=128 output.
"""
from pathlib import Path

print("THIS IS GEMMA2_05 — INT4 context K10 suppress. Not a notebook generator.")
hits = list(Path("/kaggle/input").rglob("gemma2_05_int4_context_suppress_transfer.py"))
hits += list(Path("/kaggle/working").rglob("gemma2_05_int4_context_suppress_transfer.py"))
hits = [h for h in hits if h.is_file()]
assert hits, "Add Data: kaggle_juice_gemma2_int4_context_suppress.zip"
script = hits[0]
print("SCRIPT", script)
exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"))
