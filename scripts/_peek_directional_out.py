import json
from pathlib import Path
from collections import Counter

p = Path(r"d:\tt\direction-output")
summary = json.loads((p / "directional_cell3_summary.json").read_text(encoding="utf-8"))
print(json.dumps(summary, indent=2))
print("\n--- examples ---")
for name in [
    "steer_baseline.json",
    "steer_context_k10.json",
    "steer_context_k5.json",
    "steer_parametric_k2.json",
    "steer_parametric_k3.json",
]:
    rows = json.loads((p / name).read_text(encoding="utf-8"))
    print(f"\n{name} {dict(Counter(r['label'] for r in rows))}")
    for r in rows[:5]:
        print(f"  [{r['label']}] {r['text']!r}")
