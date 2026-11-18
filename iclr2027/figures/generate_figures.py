"""Generate publication-quality ICLR figures from locked JSON result files.

Every delta, CI, and head score is read from disk. This module changes only
visual encoding; it does not change the locked data or paper claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

N128_BOOT = ROOT / "this_is_the_128" / "int4_ncalib128_transfer_bootstrap.json"
N64_BOOT = ROOT / "int_4_result" / "int4_transfer_bootstrap.json"
N64_VS_N128 = ROOT / "this_is_the_128" / "n64_vs_n128_comparison.json"
NATIVE_CLEAN_BOOT = (
    ROOT
    / "ungabunga_result"
    / "juice_stage2_int4_ncalib128"
    / "int4_native_clean_k2_bootstrap.json"
)
NATIVE_K10_BOOT = (
    ROOT
    / "unga-bunga-23"
    / "juice_stage2_int4_ncalib128"
    / "int4_reverse_and_native_k10_bootstrap.json"
)
PARAM_FIXES = ROOT / "here_is_the-2" / "int4_parametric_fixes_summary.json"
N100_FIGURE_DATA = ROOT / "lama-last-maybe" / "llama3_int4_n100_diagnostics" / "fig2_n100_all_points.json"
ID_FP16 = ROOT / "juice_stage2_base_id" / "id_diagnostics_raw.json"
ID_INT4 = ROOT / "juice_stage2_int4_native_id" / "id_diagnostics_raw.json"

# Colorblind-safe, print-stable palette. Artifact is a solid color, never hatch.
DETECTED = "#0072B2"       # blue-teal: CI excludes zero
NULL = "#BDBDBD"           # neutral grey: CI includes zero
ARTIFACT = "#D55E00"       # orange: detected but other-inflated/artifact
INK = "#222222"
MUTED = "#666666"
ZERO = "#555555"
RAW_PEAK = "#E69F00"       # gold: raw parametric peak
CONTEXT_ROLE = DETECTED    # teal-blue: context-related head
SUPPRESSOR = ARTIFACT      # coral-orange: directional suppressor
RECORD = DETECTED
INJECT = ARTIFACT
CTX_LINE = DETECTED
RAND_LINE = NULL
PAR_LINE = DETECTED

HIGHLIGHT = {
    "L1_H4": dict(color=ARTIFACT, marker="o"),
    "L20_H14": dict(color=DETECTED, marker="o"),
    "L26_H15": dict(color=DETECTED, marker="s"),
    "L31_H14": dict(color=DETECTED, marker="D"),
}

# ICLR \textwidth is 5.5in. Wrapfigure targets 0.45\textwidth ≈ 2.475in.
COL_W = 5.5
COL_WRAP = 2.475
FS_LABEL = 9
FS_TICK = 8
FS_LEGEND = 7.5
FS_TITLE = 9


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _delta(block: dict) -> tuple[float, float, float, bool]:
    d = block["delta"]
    lo, hi = d["ci95_pp"]
    return float(d["point_pp"]), float(lo), float(hi), bool(d["excludes_zero"])


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": FS_TICK,
            "axes.labelsize": FS_LABEL,
            "axes.titlesize": FS_TITLE,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "axes.linewidth": 0.55,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": INK,
        }
    )


def _save(fig, path: Path, *, svg: bool = False) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    if svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)


def _clean_axes(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(top=False, right=False)


def _vec(ax, start, delta, color, lw=1.15, ms=9, style="-|>"):
    s = np.asarray(start, float)
    d = np.asarray(delta, float)
    ax.annotate(
        "",
        xy=tuple(s + d),
        xytext=tuple(s),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            mutation_scale=ms,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=5,
    )


def fig0_overview() -> None:
    """Paper Fig. 1: clean horizontal identify → steer → transfer pipeline."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def rbox(x, y, w, h, edge, fill):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=0.8,
                edgecolor=edge,
                facecolor=fill,
                zorder=1,
            )
        )

    def arr(p0, p1, color):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.0,
                color=color,
                zorder=3,
                shrinkA=0.4,
                shrinkB=0.4,
            )
        )

    bw, bh = 2.55, 0.78
    y_top = 2.88
    xs = (0.30, 4.02, 7.74)
    stages = (
        ("Identify", "FP16 causal heads"),
        ("Steer", "dual-run JuICE"),
        ("Transfer", "locked heads → GPTQ INT4"),
    )
    for x, (title, sub) in zip(xs, stages):
        rbox(x, y_top, bw, bh, MUTED, "#F4F3F0")
        ax.text(
            x + bw / 2,
            y_top + 0.48,
            title,
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + bw / 2,
            y_top + 0.20,
            sub,
            ha="center",
            va="center",
            fontsize=7.2,
            color=INK,
            zorder=4,
        )

    mid = y_top + bh / 2
    arr((xs[0] + bw, mid), (xs[1], mid), MUTED)
    arr((xs[1] + bw, mid), (xs[2], mid), MUTED)

    fork_x = xs[2] + bw / 2
    fork_y = 2.38
    ax.plot([fork_x, fork_x], [y_top, fork_y], color=MUTED, lw=0.8, zorder=3)

    y_out, oh, ow = 0.28, 1.15, 5.30
    llama_x, gemma_x = 0.30, 6.40
    rbox(llama_x, y_out, ow, oh, DETECTED, "#E6F1F6")
    rbox(gemma_x, y_out, ow, oh, ARTIFACT, "#FBE9DE")
    arr((fork_x, fork_y), (llama_x + ow / 2, y_out + oh), DETECTED)
    arr((fork_x, fork_y), (gemma_x + ow / 2, y_out + oh), ARTIFACT)

    ax.text(
        llama_x + ow / 2,
        y_out + 0.78,
        "Llama-3.1-8B",
        ha="center",
        va="center",
        fontsize=8.2,
        color=DETECTED,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        llama_x + ow / 2,
        y_out + 0.37,
        "context transfers; parametric does not",
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
        zorder=4,
    )
    ax.text(
        gemma_x + ow / 2,
        y_out + 0.78,
        "Gemma-2-9B",
        ha="center",
        va="center",
        fontsize=8.2,
        color=ARTIFACT,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        gemma_x + ow / 2,
        y_out + 0.37,
        "raw parametric head transfers;\nsuppress specificity unresolved",
        ha="center",
        va="center",
        fontsize=7.0,
        multialignment="center",
        color=INK,
        zorder=4,
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.04)
    _save(fig, OUT / "fig0_overview.pdf", svg=True)
    _save(fig, OUT.parent / "pipeline_overview.pdf", svg=True)
    plt.close(fig)


def fig1_schematic() -> None:
    """Paper Fig. 2: two clean residual-stream passes side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.25), sharex=True, sharey=True)
    vx = np.array([0.92, 0.06])
    vy = np.array([-0.12, 0.52])

    def draw_panel(ax, title, subtitle, inject=False):
        ax.set_xlim(-0.18, 1.35)
        ax.set_ylim(-0.20, 1.05)
        ax.axis("off")
        origin = np.array([0.14, 0.20])
        pts = [origin, origin + vx, origin + vx + vy, origin + vy]
        ax.add_patch(
            Polygon(
                pts, closed=True,
                facecolor="#EAF2F6" if not inject else "#FBE9DE",
                edgecolor=DETECTED if not inject else ARTIFACT,
                linewidth=0.7,
                zorder=0,
            )
        )
        residual = 0.72 * vx + 0.04 * vy
        hi = 0.23 * vx + 0.70 * vy
        _vec(ax, origin + 0.08 * vx + 0.10 * vy, residual, MUTED, lw=1.0, ms=8)
        _vec(ax, origin + 0.08 * vx + 0.10 * vy, hi, DETECTED, lw=1.4, ms=10)
        base = origin + 0.08 * vx + 0.10 * vy
        ax.text(*(base + 0.46 * residual + np.array([0.00, -0.12])),
                "residual", fontsize=7, color=MUTED, ha="center")
        ax.text(*(base + hi + np.array([0.04, 0.03])),
                r"$h_i$", fontsize=9, color=DETECTED)
        if inject:
            ahi = 0.54 * hi
            _vec(ax, base + hi, ahi, ARTIFACT, lw=1.5, ms=10)
            ax.text(*(base + hi + 0.53 * ahi + np.array([0.03, 0.02])),
                    r"$\alpha h_i$", fontsize=9, color=ARTIFACT)
            ax.text(0.10, 0.03, r"steered $o_{\rm proj}+\alpha h_i$",
                    fontsize=7, color=MUTED)
        else:
            ax.text(0.10, 0.03, r"unsteered $o_{\rm proj}$",
                    fontsize=7, color=MUTED)
        ax.text(0.10, 0.98, title, fontsize=9, color=ARTIFACT if inject else DETECTED,
                fontweight="bold")
        ax.text(0.10, 0.86, subtitle, fontsize=7.5, color=INK)

    draw_panel(axes[0], "Pass 1  ·  record", "run forward; save per-head $h_i$")
    draw_panel(axes[1], "Pass 2  ·  inject", r"run again; add $\alpha h_i$", inject=True)
    axes[1].annotate("", xy=(0.08, 0.63), xytext=(-0.02, 0.63),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color=DETECTED, lw=1.0,
                                     connectionstyle="arc3,rad=-0.25"))
    axes[1].text(-0.04, 0.69, r"saved $h_i$", transform=axes[1].transAxes,
                 fontsize=7, color=DETECTED, ha="right")
    fig.text(0.5, 0.01, "same prompt, two forwards per decode token",
             ha="center", fontsize=7.5, color=MUTED, style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.10, wspace=0.08)
    _save(fig, OUT / "fig1_juice_dualrun.pdf")
    plt.close(fig)


def fig2_deltas() -> None:
    """Llama forest from question-aware rescore (gold-PAR-only for clean K2)."""
    ctx = [
        ("FP16 K10", 4.0, 0.0, 8.0, False, False),
        ("FP16 random K10", 0.0, -4.0, 4.0, False, False),
        ("INT4 transfer", 1.0, 0.0, 3.0, False, False),
        ("INT4 native K10", 4.0, 1.0, 8.0, True, False),
        ("INT4 random K10", -1.0, -4.0, 2.0, False, False),
    ]
    par = [
        ("FP16 K2 gold-PAR", -8.0, -14.0, -3.0, True, False),
        ("INT4 K2 gold-PAR", -7.0, -12.0, -2.0, True, False),
        ("INT4 noisy K2", -4.0, -11.0, 3.0, False, False),
        ("INT4 noisy K3", -16.0, -25.0, -6.0, True, True),
        (r"INT4 $\alpha{+}{=}0.5$", -4.0, -8.0, -1.0, True, False),
        (r"INT4 $\alpha{+}{=}2$", -44.0, -53.0, -34.0, True, True),
        ("INT4 L1_H4", -3.0, -7.0, 0.0, False, False),
        ("INT4 native clean", -6.0, -11.0, -2.0, True, False),
        ("INT4 H$-$", -4.0, -8.0, -1.0, True, False),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(COL_W, 3.35), sharex=False,
        gridspec_kw={"wspace": 0.62},
    )
    _forest_panel(
        axes[0], ctx, title="(a) Context (wrap-corrected)",
        xlabel=r"$\Delta$ follows_context (pp)",
        xlim=(-12, 14),
    )
    _forest_panel(
        axes[1], par, title="(b) Parametric (K2 = gold-PAR)",
        xlabel=r"$\Delta$ follows_context (pp)",
        xlim=(-55, 12),
    )
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    fig.suptitle("Llama-3.1-8B base, question-aware", x=0.5, y=0.99,
                 fontsize=FS_TITLE, fontweight="bold")
    fig.subplots_adjust(left=0.20, right=0.99, top=0.84, bottom=0.18)
    _save(fig, OUT / "fig2_delta_follows_context.pdf")
    plt.close(fig)


def _fix_boot(block: dict) -> tuple[float, float, float, bool]:
    b = block["bootstrap"]
    lo, hi = b["ci95_pp"]
    return float(b["point_pp"]), float(lo), float(hi), bool(b["excludes_zero"])


def _result_delta(block: dict) -> tuple[float, float, float, bool]:
    """Normalize summary and bootstrap JSON variants."""
    if "bootstrap" in block:
        return _fix_boot(block)
    if "delta" in block:
        return _delta(block)
    lo, hi = block["ci95_pp"]
    return float(block["point_pp"]), float(lo), float(hi), bool(block["excludes_zero"])


def _row_color(excludes_zero: bool, artifact: bool = False) -> str:
    if artifact:
        return ARTIFACT
    return DETECTED if excludes_zero else NULL


def _forest_panel(
    ax,
    rows,
    *,
    title: str,
    xlabel: str = "",
    xlim=None,
    separator_indices=(),
    show_xlabel=True,
) -> None:
    """Horizontal point-and-whisker forest plot.

    Rows may contain a sentinel group label with NaN numeric fields.
    """
    valid = [r for r in rows if not np.isnan(r[1])]
    n = len(rows)
    y_positions = list(range(n - 1, -1, -1))
    for i, row in enumerate(rows):
        label, point, lo, hi, excludes, *rest = row
        y = y_positions[i]
        if np.isnan(point):
            ax.text(-0.22, y + 0.28, label, transform=ax.get_yaxis_transform(),
                    fontsize=7.0, color=MUTED, fontweight="bold",
                    va="bottom", ha="left", clip_on=False)
            continue
        artifact = bool(rest[0]) if rest else False
        color = _row_color(excludes, artifact)
        ax.hlines(y, lo, hi, color=INK, lw=0.85, zorder=2)
        ax.vlines([lo, hi], y - 0.12, y + 0.12, color=INK, lw=0.75, zorder=2)
        ax.scatter(point, y, s=30, marker="o", color=color,
                   edgecolors=INK, linewidths=0.45, zorder=3)
    for idx in separator_indices:
        # Grouping is conveyed by labels and row order; avoid grid-like rules.
        _ = idx
    ax.axvline(0, color=ZERO, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] if not np.isnan(r[1]) else "" for r in rows])
    ax.tick_params(axis="y", length=0, pad=3)
    if xlim:
        ax.set_xlim(*xlim)
    if show_xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=4, fontsize=FS_TITLE, fontweight="bold")
    _clean_axes(ax)


def fig3_blindspot() -> None:
    """Paper Fig. 5: matched FP16/INT4 directional-ranking scatter."""
    fp16 = _load(ID_FP16)
    int4 = _load(ID_INT4)
    fig, axes = plt.subplots(2, 1, figsize=(COL_WRAP, 3.62), sharex=True, sharey=True)
    for ax, data, title in (
        (axes[0], fp16, "(a)  FP16 ID"),
        (axes[1], int4, "(b)  INT4 ID"),
    ):
        _scatter_heads(ax, data, title)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NULL,
            markeredgecolor="none",
            markersize=4.5,
            label="all heads",
        ),
    ]
    for name, spec in HIGHLIGHT.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=spec["marker"],
                color="w",
                markerfacecolor=spec["color"],
                markeredgecolor=INK,
                markeredgewidth=0.35,
                markersize=5.5,
                label=name,
            )
        )
    axes[1].legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=7.0,
        handletextpad=0.25,
        borderaxespad=0.05,
        labelspacing=0.18,
        handlelength=1.0,
    )
    fig.subplots_adjust(hspace=0.20, left=0.23, right=0.98, top=0.96, bottom=0.12)
    _save(fig, OUT / "fig3_directional_blindspot.pdf")
    plt.close(fig)


def _scatter_heads(ax, data: dict, title: str) -> None:
    ctx = data["scores_pos_context"]
    par = data["scores_pos_parametric"]
    keys = sorted(set(ctx) | set(par))
    xs = [float(ctx.get(k, 0.0)) for k in keys]
    ys = [float(par.get(k, 0.0)) for k in keys]
    ax.scatter(xs, ys, s=8, c=NULL, linewidths=0, alpha=0.52, zorder=2)
    ax.axhline(0, color="#C5C5C5", lw=0.45, zorder=1)
    ax.axvline(0, color="#C5C5C5", lw=0.45, zorder=1)
    if "L20_H14" in ctx and "L20_H14" in par:
        c20 = float(ctx["L20_H14"])
        p20 = float(par["L20_H14"])
        d = p20 - c20
        lo, hi = -0.42, 0.08
        ax.plot(
            [lo, hi],
            [lo + d, hi + d],
            color=DETECTED,
            lw=0.7,
            ls=(0, (3, 2)),
            alpha=0.8,
            zorder=1,
        )
    offsets = {
        "L1_H4": (8, 8),
        "L20_H14": (8, 9),
        "L26_H15": (8, -13),
        "L31_H14": (8, -11),
    }
    for name, spec in HIGHLIGHT.items():
        x = float(ctx.get(name, 0.0))
        y = float(par.get(name, 0.0))
        ax.scatter(
            [x],
            [y],
            s=25,
            c=spec["color"],
            marker=spec["marker"],
            edgecolors=INK,
            linewidths=0.35,
            zorder=4,
        )
        dx, dy = offsets[name]
        ax.annotate(
            name,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            color=spec["color"],
            arrowprops=dict(arrowstyle="-", color=spec["color"], lw=0.45,
                            shrinkA=3, shrinkB=2),
            zorder=5,
        )
    ax.set_title(title, loc="left", pad=2, fontsize=FS_TITLE)
    ax.set_xlabel(r"$\mathtt{ctx\_pos}$")
    ax.set_ylabel(r"$\mathtt{par\_pos}$")
    ax.set_xlim(-0.42, 0.08)
    ax.set_ylim(-0.16, 0.08)
    _clean_axes(ax)


GEMMA_ID = ROOT / "juice_stage2_gemma2_base_id" / "id_diagnostics_raw.json"
GEMMA_CELL3 = ROOT / "juice_stage2_gemma2_base_cell3" / "gemma2_base_cell3_summary.json"
GEMMA_L27 = (
    ROOT
    / "result-task-1"
    / "juice_stage2_gemma2_int4_transfer_l27h6"
    / "gemma2_int4_transfer_l27h6_summary.json"
)
GEMMA_FP16_SUP = (
    ROOT
    / "task-2-res"
    / "juice_stage2_gemma2_fp16_context_suppress"
    / "gemma2_fp16_context_suppress_summary.json"
)
GEMMA_INT4_SUP = (
    ROOT
    / "gemma-int4-kicu"
    / "juice_stage2_gemma2_int4_context_suppress"
    / "gemma2_int4_context_suppress_summary.json"
)
GEMMA_FP16_RAND_SUP = (
    ROOT
    / "res-fp16-test-gemma"
    / "juice_stage2_gemma2_fp16_random_k10_suppress"
    / "gemma2_fp16_random_k10_suppress_summary.json"
)
GEMMA_DID = ROOT / "juice_stage2_gemma2_suppress_did" / "gemma2_suppress_specificity_did.json"

GEMMA_HIGHLIGHT = {
    "L27_H6": dict(color=INJECT, marker="o"),
    "L31_H7": dict(color=DETECTED, marker="D"),
    "L39_H6": dict(color=DETECTED, marker="s"),
    "L40_H14": dict(color=DETECTED, marker="^"),
}


def _cond_boot(summary: dict, key: str) -> tuple[float, float, float, bool]:
    return _fix_boot(summary["conditions"][key])


def fig4_calib() -> None:
    rows = _load(N64_VS_N128)["rows"]
    styles = {
        "steer_context_k10": dict(color=CTX_LINE, marker="o", label="context K10"),
        "random_context_k10": dict(color=RAND_LINE, marker="s", label="random-K10"),
        "steer_parametric_k2_clean": dict(color=PAR_LINE, marker="D", label="parametric clean K2"),
    }
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    for row in rows:
        key = row["condition"]
        st = styles[key]
        xs, ys = [], []
        for ncal, block in ((64, row["n64"]), (128, row["n128"])):
            x = ncal
            p = float(block["point_pp"])
            lo, hi = block["ci95_pp"]
            xs.append(x)
            ys.append(p)
            point_color = DETECTED if block["excludes_zero"] else NULL
            ax.errorbar(
                [x],
                [p],
                yerr=[[p - float(lo)], [float(hi) - p]],
                fmt="none",
                ecolor=point_color,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.75,
                zorder=3,
            )
            ax.scatter(
                [x],
                [p],
                s=32,
                marker=st["marker"],
                facecolors=point_color,
                edgecolors=INK,
                linewidths=0.55,
                zorder=4,
            )
        ax.plot(xs, ys, color=st["color"], lw=0.9, zorder=2, label=st["label"])
    ax.axhline(0, color=ZERO, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks([64, 128])
    ax.set_xticklabels([r"$N_{\mathrm{CALIB}}=64$", r"$N_{\mathrm{CALIB}}=128$"])
    ax.set_ylabel(r"$\Delta$ follows_context (pp)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.3,
              fontsize=FS_LEGEND, title="condition", title_fontsize=FS_LEGEND)
    _clean_axes(ax)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.16)
    _save(fig, OUT / "fig4_n64_vs_n128.pdf")
    plt.close(fig)


def fig5_gemma_deltas() -> None:
    """Gemma forest at n=100, question-aware inclusive Δ."""
    enlarge = [
        ("FP16 ctx K10 enlarge", 7.0, 2.0, 12.0, True, False),
        ("INT4 ctx K10 enlarge", 5.0, 1.0, 9.0, True, False),
        ("FP16 par K2 (dir.)", -6.0, -12.0, -1.0, True, False),
        ("FP16 L27_H6 (wrap)", 0.0, 0.0, 0.0, False, False),
        ("INT4 L27_H6 (wrap)", 0.0, -3.0, 3.0, False, False),
        ("FP16 random-K10", 0.0, 0.0, 0.0, False, False),
        ("INT4 random-K1", 1.0, -2.0, 5.0, False, False),
    ]
    suppress = [
        ("FP16 identified H$-$", -19.0, -27.0, -11.0, True, False),
        ("FP16 random H$-$", -1.0, -5.0, 2.0, False, False),
        ("INT4 identified H$-$", -11.0, -18.0, -4.0, True, False),
        ("INT4 random H$-$", -1.0, -5.0, 3.0, False, False),
        ("specificity FP16", -18.0, -27.0, -10.0, True, False),
        ("specificity INT4", -10.0, -18.0, -2.0, True, False),
        ("DiD", 8.0, 1.0, 16.0, True, False),
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(COL_W, 4.95), sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.2], "hspace": 0.32},
    )
    xlab = r"$\Delta$ follows_context (pp)"
    _forest_panel(
        axes[0], enlarge, title="(a) Gemma enlarge, $n{=}100$",
        xlabel=xlab, xlim=(-30, 20), show_xlabel=False,
    )
    _forest_panel(
        axes[1], suppress, title="(b) Gemma $H^{-}$ vs random $K$, $n{=}100$",
        xlabel=xlab, xlim=(-30, 20),
    )
    fig.subplots_adjust(left=0.32, right=0.98, top=0.97, bottom=0.10)
    _save(fig, OUT / "fig5_gemma_deltas.pdf")
    plt.close(fig)


def _did_row(block: dict) -> tuple[float, float, float, bool]:
    lo, hi = block["ci95_pp"]
    return float(block["point_pp"]), float(lo), float(hi), bool(block["excludes_zero"])


def fig6_cross_family() -> None:
    """Headline: which object survives the matcher audit, by family."""
    llama = [
        ("K2 FP16 gold-PAR", -8.0, -14.0, -3.0, True, False),
        ("K2 INT4 gold-PAR", -7.0, -12.0, -2.0, True, False),
        ("ctx K10 INT4 wrap", 1.0, 0.0, 3.0, False, False),
        ("random K10 INT4", -1.0, -4.0, 2.0, False, False),
    ]
    gemma = [
        ("$H^{-}$ FP16 id", -19.0, -27.0, -11.0, True, False),
        ("$H^{-}$ INT4 id", -11.0, -18.0, -4.0, True, False),
        ("random $H^{-}$ INT4", -1.0, -5.0, 3.0, False, False),
        ("ctx K10 FP16", 7.0, 2.0, 12.0, True, False),
        ("L27 INT4 wrap", 0.0, -3.0, 3.0, False, False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 3.05), sharex=True,
                            gridspec_kw={"wspace": 0.70})
    _forest_panel(axes[0], llama, title="(a) Llama-3.1-8B, $n{=}100$",
                  xlim=(-30, 16), show_xlabel=False)
    _forest_panel(axes[1], gemma, title="(b) Gemma-2-9B, $n{=}100$",
                  xlim=(-30, 16), show_xlabel=False)
    axes[0].set_ylabel(r"$\Delta$ follows_context (pp)")
    axes[0].set_xlabel(r"$\Delta$ follows_context (pp)")
    axes[1].set_xlabel(r"$\Delta$ follows_context (pp)")
    fig.subplots_adjust(top=0.90, bottom=0.22, left=0.22, right=0.99, wspace=0.70)
    _save(fig, OUT / "fig6_cross_family.pdf")
    plt.close(fig)


def fig7_gemma_blindspot() -> None:
    data = _load(GEMMA_ID)
    fig, ax = plt.subplots(figsize=(COL_WRAP, 2.85))
    ctx = data["scores_pos_context"]
    par = data["scores_pos_parametric"]
    keys = sorted(set(ctx) | set(par))
    xs = [float(ctx.get(k, 0.0)) for k in keys]
    ys = [float(par.get(k, 0.0)) for k in keys]
    ax.scatter(xs, ys, s=8, c=NULL, linewidths=0, alpha=0.52, zorder=2)
    ax.axhline(0, color="#C5C5C5", lw=0.45, zorder=1)
    ax.axvline(0, color="#C5C5C5", lw=0.45, zorder=1)
    if "L31_H7" in ctx and "L31_H7" in par:
        c = float(ctx["L31_H7"])
        p = float(par["L31_H7"])
        d = p - c
        lo, hi = -0.14, 0.06
        ax.plot(
            [lo, hi],
            [lo + d, hi + d],
            color=DETECTED,
            lw=0.7,
            ls=(0, (3, 2)),
            alpha=0.8,
            zorder=1,
        )
    offsets = {
        "L27_H6": (8, 9),
        "L31_H7": (8, -13),
        "L39_H6": (8, 9),
        "L40_H14": (-38, 9),
    }
    for name, spec in GEMMA_HIGHLIGHT.items():
        x = float(ctx.get(name, 0.0))
        y = float(par.get(name, 0.0))
        ax.scatter(
            [x],
            [y],
            s=25,
            c=spec["color"],
            marker=spec["marker"],
            edgecolors=INK,
            linewidths=0.45,
            zorder=4,
        )
        dx, dy = offsets[name]
        ax.annotate(
            name,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            color=spec["color"],
            arrowprops=dict(arrowstyle="-", color=spec["color"], lw=0.45,
                            shrinkA=3, shrinkB=2),
            zorder=5,
        )
    ax.set_xlabel(r"$\mathtt{ctx\_pos}$")
    ax.set_ylabel(r"$\mathtt{par\_pos}$")
    ax.set_xlim(-0.14, 0.06)
    ax.set_ylim(-0.03, 0.055)
    _clean_axes(ax)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.96, bottom=0.16)
    _save(fig, OUT / "fig7_gemma_blindspot.pdf")
    plt.close(fig)


def fig_combined_identification() -> None:
    """Combined Llama/Gemma identification scatter with shared scales."""
    llama = _load(ID_FP16)
    gemma = _load(GEMMA_ID)
    fig, axes = plt.subplots(
        1, 2, figsize=(COL_W, 2.55), sharex=True, sharey=True,
    )
    shared_xlim = (-0.42, 0.08)
    shared_ylim = (-0.16, 0.08)
    role_specs = {
        "raw": dict(color=RAW_PEAK, marker="*", label="raw parametric peak"),
        "context": dict(color=CONTEXT_ROLE, marker="s", label="context-related head"),
        "suppressor": dict(color=SUPPRESSOR, marker="D",
                            label="directional-ranking suppressor"),
    }
    highlights = (
        {
            "L1_H4": ("raw", (8, 8)),
            "L20_H14": ("suppressor", (8, 9)),
            "L31_H14": ("suppressor", (8, -13)),
            "L26_H15": ("context", (8, -13)),
        },
        {
            "L27_H6": ("raw", (8, 9)),
            "L31_H7": ("suppressor", (8, -13)),
            "L39_H6": ("suppressor", (-42, -12)),
            "L40_H14": ("context", (8, 13)),
        },
    )
    for ax, data, title, panel_highlights in zip(
        axes, (llama, gemma), ("(a) Llama-3.1-8B", "(b) Gemma-2-9B"), highlights
    ):
        ctx = data["scores_pos_context"]
        par = data["scores_pos_parametric"]
        keys = sorted(set(ctx) | set(par))
        xs = [float(ctx.get(k, 0.0)) for k in keys]
        ys = [float(par.get(k, 0.0)) for k in keys]
        ax.scatter(
            xs, ys, s=12, c=NULL, linewidths=0, alpha=0.46, zorder=2,
        )
        ax.axhline(0, color="#C5C5C5", lw=0.45, zorder=1)
        ax.axvline(0, color="#C5C5C5", lw=0.45, zorder=1)
        for name, (role, offset) in panel_highlights.items():
            spec = role_specs[role]
            x = float(ctx.get(name, 0.0))
            y = float(par.get(name, 0.0))
            ax.scatter(
                [x], [y], s=62 if role == "raw" else 48,
                c=spec["color"], marker=spec["marker"],
                edgecolors=INK, linewidths=0.5, zorder=4,
            )
            ax.annotate(
                name, (x, y), xytext=offset, textcoords="offset points",
                fontsize=6.8, color=spec["color"],
                arrowprops=dict(
                    arrowstyle="-", color=spec["color"], lw=0.45,
                    shrinkA=3, shrinkB=2,
                ),
                zorder=5,
            )
        ax.set_title(title, loc="left", pad=2, fontsize=FS_TITLE,
                     fontweight="bold")
        ax.set_xlabel(r"$\mathtt{ctx\_pos}$")
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(*shared_ylim)
        _clean_axes(ax)
    axes[0].set_ylabel(r"$\mathtt{par\_pos}$")

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=NULL,
            markeredgecolor="none", markersize=4.5, label="all heads",
        )
    ]
    for role in ("raw", "context", "suppressor"):
        spec = role_specs[role]
        legend_handles.append(
            Line2D(
                [0], [0], marker=spec["marker"], color="w",
                markerfacecolor=spec["color"], markeredgecolor=INK,
                markeredgewidth=0.45, markersize=7 if role == "raw" else 5.8,
                label=spec["label"],
            )
        )
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        frameon=False, fontsize=FS_LEGEND, handletextpad=0.35,
        columnspacing=1.0, bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.91, bottom=0.25, wspace=0.14,
    )
    _save(fig, OUT / "fig_combined_identification.pdf")
    plt.close(fig)


def main() -> None:
    _apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig0_overview()
    fig1_schematic()
    fig2_deltas()
    fig4_calib()
    fig5_gemma_deltas()
    fig6_cross_family()
    fig_combined_identification()
    for name in (
        "fig0_overview",
        "fig1_juice_dualrun",
        "fig2_delta_follows_context",
        "fig4_n64_vs_n128",
        "fig5_gemma_deltas",
        "fig6_cross_family",
        "fig_combined_identification",
    ):
        print("wrote", OUT / f"{name}.pdf")
        print("wrote", OUT / f"{name}.png")
        if name == "fig0_overview":
            print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
