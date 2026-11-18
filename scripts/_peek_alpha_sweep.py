import json
import re
from pathlib import Path
from collections import Counter

p = Path(r"d:\tt\juice_stage2_alpha_sweep")


def coherent(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if t.count("://") >= 3:
        return False
    if re.fullmatch(r"[\W\d_]+", t):
        return False
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= max(2, int(0.3 * len(t)))


files = [
    "steer_baseline.json",
    "steer_context_ap1p0_am0p0.json",
    "steer_parametric_ap1p0_am0p0.json",
    "steer_context_ap2p0_am1p0.json",
    "steer_parametric_ap2p0_am1p0.json",
]

for name in files:
    rows = json.loads((p / name).read_text(encoding="utf-8"))
    labs = Counter(r["label"] for r in rows)
    fc = sum(1 for r in rows if r.get("follows_context") or r["label"] == "context") / len(rows)
    coh = sum(1 for r in rows if coherent(r["text"])) / len(rows)
    print(f"=== {name} === fc={fc:.3f} coh={coh:.3f} {dict(labs)}")
    for r in rows[:5]:
        print(f"  [{r['label']}] {r['text']!r}")
    print()
