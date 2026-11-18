import json
from pathlib import Path
from collections import Counter

p = Path(r"c:\Users\Tanjid\Downloads\results\juice_stage2_fp16")

for name in ["heads_context.json", "heads_parametric.json"]:
    d = json.loads((p / name).read_text(encoding="utf-8"))
    print("===", name, "===")
    print("K", d.get("K"), "n_calib", d.get("n_calibration"), "target", d.get("steer_target"))
    print("enlarge", [f"L{h['layer']}_H{h['head']}" for h in d["enlarge"]])
    print("suppress", [f"L{h['layer']}_H{h['head']}" for h in d["suppress"]])
    print("top5 pos", list(d["scores_pos"].items())[:5])
    print("top5 neg", list(d["scores_neg"].items())[:5])
    print()

for name in ["steer_baseline.json", "steer_context.json", "steer_parametric.json"]:
    rows = json.loads((p / name).read_text(encoding="utf-8"))
    print("===", name, "===", dict(Counter(r["label"] for r in rows)))
    for r in rows[:8]:
        t = r["text"].replace("\n", "\\n")
        print(f"  {r['id']}: [{r['label']}] {t[:120]!r}")
    print("  ... lens", sorted(len(r["text"]) for r in rows)[:5], "...", sorted(len(r["text"]) for r in rows)[-3:])
    print()
