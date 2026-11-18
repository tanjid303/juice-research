"""Compare N_CALIB=64 vs N_CALIB=128 Stage 2 INT4 transfer CIs.

Table columns (locked names from int4_transfer_bootstrap.json / int4_random_k_summary.json):
  condition | N64 Δ [CI] | N128 Δ [CI] | same sign? | does "excludes 0" verdict flip?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

STAGE_T0 = time.time()


def elapsed(stage: str) -> None:
    print(f"[elapsed] {stage}: {(time.time() - STAGE_T0) / 60:.1f} min since start")


def _find_file(name: str, prefer_substrings: tuple[str, ...] = ()) -> Path:
    hits: list[Path] = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input"), Path(r"d:\tt")):
        if not root.exists():
            continue
        try:
            hits.extend(p for p in root.rglob(name) if p.is_file())
        except OSError:
            pass
    assert hits, f"Missing {name}"
    def score(p: Path) -> tuple:
        s = str(p).replace("\\", "/").lower()
        pref = 0
        for i, sub in enumerate(prefer_substrings):
            if sub.lower() in s:
                pref -= 10 - i
        want_n128 = any("ncalib128" in x.lower() for x in prefer_substrings)
        want_n64 = any(x in prefer_substrings for x in ("n64_baseline", "int_4_result", "here_is_the"))
        if want_n128 and ("n64_baseline" in s or "int_4_result" in s or "here_is_the" in s):
            pref += 100
        if want_n64 and "ncalib128" in s:
            pref += 100
        return (pref, len(s))
    hits.sort(key=score)
    print(name, "prefer", prefer_substrings, "->", hits[0])
    return hits[0]


def _delta_block(obj: dict) -> dict:
    """Normalize to the `delta` object from bootstrap_paired_follow_delta."""
    if "delta" in obj and isinstance(obj["delta"], dict) and "point_pp" in obj["delta"]:
        return obj["delta"]
    if "bootstrap" in obj and isinstance(obj["bootstrap"], dict) and "point_pp" in obj["bootstrap"]:
        return obj["bootstrap"]
    if "point_pp" in obj:
        return obj
    raise KeyError(f"no delta/point_pp in keys={list(obj)[:12]}")


def _sign(x: float) -> str:
    if x > 0:
        return "+"
    if x < 0:
        return "−"
    return "0"


def _fmt_delta(d: dict) -> str:
    lo, hi = d["ci95_pp"]
    return f"{d['point_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def main() -> None:
    out_dir = Path("/kaggle/working/juice_stage2_int4_ncalib128")
    if not out_dir.exists():
        out_dir = Path(r"d:\tt") / "juice_stage2_int4_ncalib128"
    out_dir.mkdir(parents=True, exist_ok=True)

    n64_boot_path = _find_file(
        "int4_transfer_bootstrap.json",
        prefer_substrings=("n64_baseline", "int_4_result"),
    )
    n64_rand_path = _find_file(
        "int4_random_k_summary.json",
        prefer_substrings=("n64_baseline", "here_is_the"),
    )
    n128_boot_path = _find_file(
        "int4_transfer_bootstrap.json",
        prefer_substrings=("int4_ncalib128", "ncalib128"),
    )
    n128_rand_path = None
    try:
        n128_rand_path = _find_file(
            "int4_random_k_summary.json",
            prefer_substrings=("int4_ncalib128", "ncalib128"),
        )
    except AssertionError:
        pass

    n64_boot = json.loads(n64_boot_path.read_text(encoding="utf-8"))
    n64_rand = json.loads(n64_rand_path.read_text(encoding="utf-8"))
    n128_boot = json.loads(n128_boot_path.read_text(encoding="utf-8"))
    if n64_boot_path.resolve() == n128_boot_path.resolve():
        raise FileNotFoundError(
            "N128 bootstrap resolved to the same file as N64. "
            "Run script 02 first so juice_stage2_int4_ncalib128/int4_transfer_bootstrap.json exists."
        )
    n128_rand = (
        json.loads(n128_rand_path.read_text(encoding="utf-8")) if n128_rand_path else None
    )

    assert n64_boot["n_boot"] == 1000 and n64_boot["seed"] == 42
    assert n128_boot["n_boot"] == 1000 and n128_boot["seed"] == 42

    n64_ctx = _delta_block(n64_boot["int4"]["conditions"]["steer_context_k10"])
    n64_par = _delta_block(n64_boot["int4"]["conditions"]["steer_parametric_k2_clean"])
    n64_rnd = _delta_block(n64_rand["conditions"]["random_context_k10"])

    n128_ctx = _delta_block(n128_boot["int4"]["conditions"]["steer_context_k10"])
    n128_par = _delta_block(n128_boot["int4"]["conditions"]["steer_parametric_k2_clean"])
    if "random_context_k10" in n128_boot["int4"]["conditions"]:
        n128_rnd = _delta_block(n128_boot["int4"]["conditions"]["random_context_k10"])
    elif n128_rand is not None:
        n128_rnd = _delta_block(n128_rand["conditions"]["random_context_k10"])
    else:
        raise KeyError("N128 random_context_k10 missing from bootstrap and random-K summary")

    rows = [
        ("steer_context_k10", n64_ctx, n128_ctx),
        ("random_context_k10", n64_rnd, n128_rnd),
        ("steer_parametric_k2_clean", n64_par, n128_par),
    ]

    table = []
    print()
    print(
        f"{'condition':28s}  {'N64 Δ [CI]':22s}  {'N128 Δ [CI]':22s}  "
        f"{'same sign?':10s}  {'excl0 flip?':12s}"
    )
    print("-" * 110)
    for name, a, b in rows:
        same_sign = _sign(a["point_pp"]) == _sign(b["point_pp"])
        flip = bool(a["excludes_zero"]) != bool(b["excludes_zero"])
        rec = {
            "condition": name,
            "n64": {
                "point_pp": a["point_pp"],
                "ci95_pp": a["ci95_pp"],
                "excludes_zero": a["excludes_zero"],
                "n_draws_lt0": a.get("n_draws_lt0"),
                "n_draws_eq0": a.get("n_draws_eq0"),
                "n_draws_gt0": a.get("n_draws_gt0"),
            },
            "n128": {
                "point_pp": b["point_pp"],
                "ci95_pp": b["ci95_pp"],
                "excludes_zero": b["excludes_zero"],
                "n_draws_lt0": b.get("n_draws_lt0"),
                "n_draws_eq0": b.get("n_draws_eq0"),
                "n_draws_gt0": b.get("n_draws_gt0"),
            },
            "same_sign": same_sign,
            "excludes_zero_verdict_flip": flip,
        }
        table.append(rec)
        print(
            f"{name:28s}  {_fmt_delta(a):22s}  {_fmt_delta(b):22s}  "
            f"{'yes' if same_sign else 'NO':10s}  {'YES' if flip else 'no':12s}"
        )

    payload = {
        "n64_source": {
            "bootstrap": str(n64_boot_path),
            "random_k": str(n64_rand_path),
            "n_calib": 64,
        },
        "n128_source": {
            "bootstrap": str(n128_boot_path),
            "random_k": str(n128_rand_path) if n128_rand_path else None,
            "n_calib": 128,
        },
        "n_boot": 1000,
        "seed": 42,
        "note": (
            "Δ = steered follows_context − baseline, paired on the same 50 eval ids. "
            "same_sign compares sign(point_pp). excludes_zero_verdict_flip is True iff "
            "the N64 CI-excludes-0 call differs from the N128 call."
        ),
        "rows": table,
    }
    out_path = out_dir / "n64_vs_n128_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nwrote", out_path)
    elapsed("compare done")


if __name__ == "__main__":
    main()
