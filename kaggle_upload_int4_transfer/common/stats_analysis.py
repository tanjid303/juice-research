"""MR★ bootstrap CIs and logistic mixed-effects AME (§5.4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def shared_double_filter_ids(df: pd.DataFrame) -> set[str]:
    fp16_ok = set(df.loc[(df["precision"] == "fp16") & df["closed_book_correct"], "id"])
    int4_ok = set(df.loc[(df["precision"] == "int4") & df["closed_book_correct"], "id"])
    return fp16_ok & int4_ok


def _mr_star_on_ids(df: pd.DataFrame, precision: str, ids: list[str]) -> float:
    if not ids:
        return float("nan")
    sub = df[(df["precision"] == precision) & df["id"].isin(ids)]
    # Preserve one row per id (should already be unique per precision)
    sub = sub.drop_duplicates(subset=["id"], keep="first")
    # Align to requested id order for bootstrap stability
    by_id = {r["id"]: r["conflict_label"] for _, r in sub.iterrows()}
    labels = [by_id[i] for i in ids if i in by_id]
    if not labels:
        return float("nan")
    return sum(1 for lab in labels if lab == "parametric") / len(labels)


def bootstrap_mr_star(
    df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Resample the shared FP16∩INT4 double-filtered item set with replacement.
    Returns point estimates and 95% percentile CIs for MR★_FP16, MR★_INT4, and delta.
    """
    shared = sorted(shared_double_filter_ids(df))
    n = len(shared)
    if n == 0:
        return {"n_shared": 0, "error": "empty shared double-filter set"}

    rng = np.random.default_rng(seed)
    fp_point = _mr_star_on_ids(df, "fp16", shared)
    i4_point = _mr_star_on_ids(df, "int4", shared)
    delta_point = i4_point - fp_point

    fp_boots, i4_boots, deltas = [], [], []
    for _ in range(n_boot):
        draw = [shared[i] for i in rng.integers(0, n, size=n)]
        fp = _mr_star_on_ids(df, "fp16", draw)
        i4 = _mr_star_on_ids(df, "int4", draw)
        fp_boots.append(fp)
        i4_boots.append(i4)
        deltas.append(i4 - fp)

    def ci(xs: list[float]) -> tuple[float, float]:
        return float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))

    fp_lo, fp_hi = ci(fp_boots)
    i4_lo, i4_hi = ci(i4_boots)
    d_lo, d_hi = ci(deltas)
    return {
        "n_shared": n,
        "n_boot": n_boot,
        "seed": seed,
        "fp16_mr_star": {"point": fp_point, "ci95": [fp_lo, fp_hi]},
        "int4_mr_star": {"point": i4_point, "ci95": [i4_lo, i4_hi]},
        "delta_int4_minus_fp16": {
            "point": delta_point,
            "ci95": [d_lo, d_hi],
            "excludes_zero": not (d_lo <= 0.0 <= d_hi),
            "point_pp": 100.0 * delta_point,
            "ci95_pp": [100.0 * d_lo, 100.0 * d_hi],
        },
    }


def logistic_mixed_ame(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Primary §5.4 estimand: AME of INT4 vs FP16 on follows_context,
    from logistic mixed-effects follows_context ~ precision + (1|item)
    on the shared double-filtered set.

    follows_context = 1 iff conflict_label == 'context'.
    Falls back to cluster-robust logit AME if the mixed model fails to fit.
    """
    shared = shared_double_filter_ids(df)
    sub = df[df["id"].isin(shared) & df["precision"].isin(["fp16", "int4"])].copy()
    if sub.empty:
        return {"n_shared": 0, "error": "empty shared double-filter set"}

    sub["follows_context"] = (sub["conflict_label"] == "context").astype(int)
    sub["precision_int4"] = (sub["precision"] == "int4").astype(int)
    # Numeric item codes for statsmodels
    item_codes, _ = pd.factorize(sub["id"])
    sub["item_code"] = item_codes

    result: dict[str, Any] = {
        "n_shared": len(shared),
        "n_rows": len(sub),
        "follows_context_def": "conflict_label == 'context'",
        "formula": "follows_context ~ precision_int4 + (1|item)",
    }

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        model = BinomialBayesMixedGLM.from_formula(
            "follows_context ~ precision_int4",
            {"item": "0 + C(item_code)"},
            sub,
        )
        fit = model.fit_vb(verbose=False)
        fe = np.asarray(fit.fe_mean).reshape(-1)
        b0, b1 = float(fe[0]), float(fe[1])
        fe_sd = np.asarray(getattr(fit, "fe_sd", [np.nan, np.nan])).reshape(-1)

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

        # Population-averaged AME at RE=0 (fixed-effect only)
        ame_re0 = float(sigmoid(b0 + b1) - sigmoid(b0))
        # Item-averaged AME using fitted random intercepts
        u = np.asarray(fit.vc_mean).reshape(-1)
        ame_item = float(np.mean(sigmoid(b0 + b1 + u) - sigmoid(b0 + u)))

        s0, s1 = sigmoid(b0), sigmoid(b0 + b1)
        g = np.array([s1 * (1 - s1) - s0 * (1 - s0), s1 * (1 - s1)])
        se = float(np.sqrt((g[0] * fe_sd[0]) ** 2 + (g[1] * fe_sd[1]) ** 2)) if fe_sd.size >= 2 else float("nan")
        z = 1.96
        lo, hi = ame_re0 - z * se, ame_re0 + z * se

        result.update(
            {
                "method": "BinomialBayesMixedGLM_VB",
                "ame_follows_context": ame_re0,
                "ame_pp": 100.0 * ame_re0,
                "ame_item_averaged": ame_item,
                "ame_item_averaged_pp": 100.0 * ame_item,
                "ame_se": se,
                "ame_ci95": [lo, hi],
                "ame_ci95_pp": [100.0 * lo, 100.0 * hi],
                "ci_excludes_zero": (not np.isnan(lo)) and not (lo <= 0.0 <= hi),
                "fe_intercept": b0,
                "fe_precision_int4": b1,
                "fe_precision_int4_sd": float(fe_sd[1]) if fe_sd.size >= 2 else None,
            }
        )
    except Exception as e:
        result["mixed_model_error"] = repr(e)

    # Always also report cluster-robust logit AME (frequentist companion).
    try:
        import statsmodels.formula.api as smf

        logit = smf.logit("follows_context ~ precision_int4", data=sub).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": sub["item_code"]}
        )
        marg = logit.get_margeff(at="overall")
        ame_c = float(marg.margeff[0])
        se_c = float(marg.margeff_se[0])
        lo_c, hi_c = ame_c - 1.96 * se_c, ame_c + 1.96 * se_c
        result["cluster_robust_logit"] = {
            "ame_follows_context": ame_c,
            "ame_pp": 100.0 * ame_c,
            "ame_se": se_c,
            "ame_ci95": [lo_c, hi_c],
            "ame_ci95_pp": [100.0 * lo_c, 100.0 * hi_c],
            "ci_excludes_zero": not (lo_c <= 0.0 <= hi_c),
        }
        if "ame_pp" not in result:
            result.update(
                {
                    "method": "logit_cluster_robust_fallback",
                    "ame_follows_context": ame_c,
                    "ame_pp": 100.0 * ame_c,
                    "ame_se": se_c,
                    "ame_ci95": [lo_c, hi_c],
                    "ame_ci95_pp": [100.0 * lo_c, 100.0 * hi_c],
                    "ci_excludes_zero": not (lo_c <= 0.0 <= hi_c),
                }
            )
    except Exception as e2:
        result["fallback_error"] = repr(e2)
    return result


def analyze_behavioral(df: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    boot = bootstrap_mr_star(df, n_boot=n_boot, seed=seed)
    ame = logistic_mixed_ame(df, seed=seed)
    return {"bootstrap_mr_star": boot, "logistic_mixed_ame": ame}
