"""Draft ICLR displays for main2.tex (Steps 5-8). Frozen claim_map.md numbers only."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_figures import (  # noqa: E402
    ARTIFACT,
    COL_W,
    DETECTED,
    FS_LEGEND,
    FS_TITLE,
    INK,
    MUTED,
    NULL,
    OUT,
    ZERO,
    _apply_style,
    _clean_axes,
    _forest_panel,
    _save,
)

RECORD = "#0072B2"
INJECT = "#D55E00"


def _box(ax, x, y, w, h, edge, fill, title, sub, title_size=8.0, sub_size=6.4):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.75,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=INK,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        sub,
        ha="center",
        va="center",
        fontsize=sub_size,
        color=MUTED,
        zorder=3,
        linespacing=1.15,
    )


def _arr(ax, p0, p1, color=MUTED, linestyle="-", linewidth=0.9, mutation_scale=8):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            zorder=4,
            shrinkA=0.5,
            shrinkB=0.5,
        )
    )


def fig_pipeline() -> None:
    """New Figure 1: arbitration + identify-lock-transfer + JuICE + audit."""
    fig, ax = plt.subplots(figsize=(COL_W, 4.15))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    ax.text(0.15, 10.28, "A  Conflict", fontsize=7.4, color=MUTED, fontweight="bold")
    _box(ax, 0.15, 8.55, 3.55, 1.55, MUTED, "#F4F3F0", "Conflict prompt",
         "context entity vs\nparametric entity")
    _box(ax, 4.00, 8.55, 3.90, 1.55, DETECTED, "#E6F1F6", "Arbitration",
         "which source wins\nat decode?")
    _box(ax, 8.20, 8.55, 3.80, 1.55, ARTIFACT, "#FBE9DE", "Causal object",
         "attention heads that\nmove the winner")
    _arr(ax, (3.70, 9.32), (4.00, 9.32))
    _arr(ax, (7.90, 9.32), (8.20, 9.32))

    ax.text(0.15, 8.18, "B  Identify → lock → transfer",
            fontsize=7.4, color=MUTED, fontweight="bold")
    _box(ax, 0.15, 6.28, 3.55, 1.70, DETECTED, "#E6F1F6", "Identify on FP16",
         "directional ctx vs par\ntop-$K$ heads")
    _box(ax, 4.20, 6.28, 3.50, 1.70, INK, "#F4F3F0", "Lock the set",
         "same heads, same $\\alpha$\nacross precisions")
    _box(ax, 8.20, 6.28, 3.80, 1.70, ARTIFACT, "#FBE9DE", "Transfer to INT4",
         "apply locked set on\nGPTQ INT4")
    _arr(ax, (3.70, 7.13), (4.20, 7.13), DETECTED)
    _arr(ax, (7.70, 7.13), (8.20, 7.13), ARTIFACT)

    ax.text(0.15, 5.90, "C  Dual-run JuICE (decode-time, not fine-tune)",
            fontsize=7.4, color=MUTED, fontweight="bold")
    _box(ax, 0.15, 4.05, 5.55, 1.65, RECORD, "#E6F1F6", "Pass 1  ·  record",
         "unsteered forward; save h_i after o_proj")
    _box(ax, 6.50, 4.05, 5.50, 1.65, INJECT, "#FBE9DE", "Pass 2  ·  inject",
         "o_proj += α h_i; steered tokens")
    _arr(ax, (5.70, 4.88), (6.50, 4.88), RECORD)
    ax.text(6.10, 5.18, r"$\alpha \times h_i$", ha="center", fontsize=6.2, color=RECORD)

    ax.text(0.15, 3.68, "D  Question-aware / gold-PAR audit",
            fontsize=7.4, color=MUTED, fontweight="bold")
    y_d = 0.36
    h_d = 3.07
    arrow_y = y_d + h_d / 2

    w_box = 2.76
    gap_solid = 0.28
    x0 = 0.15
    x1 = x0 + w_box + gap_solid
    x2 = x1 + w_box + gap_solid
    gap_dash = 0.40
    x3 = x2 + w_box + gap_dash
    w_ctrl = 12.00 - x3

    _box(ax, x0, y_d, w_box, h_d, MUTED, "#F4F3F0", "Original matcher",
         "parametric-first\nrole frames can\nscore a wrap as\nparametric")
    _box(ax, x1, y_d, w_box, h_d, DETECTED, "#E6F1F6", "Question-aware",
         "entity must answer\nthe question; wraps\nare not CTX/PAR\nswitches")
    _box(ax, x2, y_d, w_box, h_d, ARTIFACT, "#FBE9DE", "Gold-PAR-only",
         "leave-context counts\nonly if gold\nparametric entity\nis named")
    _box(ax, x3, y_d, w_ctrl, h_d, NULL, "#F0F0F0", "Controls",
         "random $K$, same\ncardinality")

    _arr(ax, (x0 + w_box, arrow_y), (x1, arrow_y))
    _arr(ax, (x1 + w_box, arrow_y), (x2, arrow_y))
    _arr(ax, (x2 + w_box, arrow_y), (x3, arrow_y), color="#8E8E8E", linestyle=(0, (3, 2)), linewidth=0.85, mutation_scale=7.5)

    ax.text(x3 + w_ctrl / 2, 0.14, "matched cells only", ha="center", va="center",
            fontsize=6.2, color="#666666", fontstyle="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    _save(fig, OUT / "fig_pipeline.pdf")
    plt.close(fig)


def fig_llama() -> None:
    """New Figure 2: Llama wrap audit (A) and clean K2 (B)."""
    ctx = [
        ("FP16 original K10", 8.0, 2.0, 14.0, True, True),
        ("FP16 question-aware K10", 4.0, 0.0, 8.0, False, False),
        ("FP16 random K10", 0.0, -4.0, 4.0, False, False),
        ("INT4 original K10", 15.0, 8.0, 23.0, True, True),
        ("INT4 question-aware K10", 1.0, 0.0, 3.0, False, False),
        ("INT4 random K10", -1.0, -4.0, 2.0, False, False),
    ]
    k2 = [
        ("FP16 inclusive QA", -15.0, -22.0, -7.0, True, False),
        ("FP16 gold-PAR-only", -8.0, -14.0, -3.0, True, False),
        ("FP16 random K2", -1.0, -3.0, 0.0, False, False),
        ("INT4 inclusive QA", -12.0, -19.0, -6.0, True, False),
        ("INT4 gold-PAR-only", -7.0, -12.0, -2.0, True, False),
        ("INT4 random K2", -1.0, -3.0, 0.0, False, False),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(COL_W, 3.55),
        gridspec_kw={"wspace": 0.55, "width_ratios": [1.08, 0.92]},
    )
    xlab = r"$\Delta$ follows_context (pp)"
    _forest_panel(
        axes[0], ctx, title="A  Context K10, n=100  ·  17/17 INT4 wraps",
        xlabel=xlab, xlim=(-10, 26),
    )
    _forest_panel(
        axes[1], k2, title="B  Clean K2, n=100  ·  confirms specificity",
        xlabel=xlab, xlim=(-28, 4),
    )
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ARTIFACT,
               markeredgecolor=INK, markeredgewidth=0.45, markersize=6,
               label="detected (original / wrap-prone)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DETECTED,
               markeredgecolor=INK, markeredgewidth=0.45, markersize=6,
               label="detected (audited)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NULL,
               markeredgecolor=INK, markeredgewidth=0.45, markersize=6,
               label="null (CI includes 0)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        fontsize=FS_LEGEND, bbox_to_anchor=(0.5, -0.01),
        handletextpad=0.35, columnspacing=0.9,
    )
    fig.suptitle("Llama-3.1-8B base", x=0.5, y=0.995, fontsize=FS_TITLE, fontweight="bold")
    fig.subplots_adjust(left=0.26, right=0.98, top=0.86, bottom=0.18)
    _save(fig, OUT / "fig_llama_audit.pdf")
    plt.close(fig)


def fig_gemma() -> None:
    """New Figure 3: Gemma H−, enlarge, L27 — attenuation and overlap visible."""
    hminus = [
        ("FP16 identified $H^{-}$", -19.0, -27.0, -11.0, True, False),
        ("FP16 random $H^{-}$", -1.0, -5.0, 2.0, False, False),
        ("FP16 specificity", -18.0, -27.0, -10.0, True, False),
        ("INT4 identified $H^{-}$", -11.0, -18.0, -4.0, True, False),
        ("INT4 random $H^{-}$", -1.0, -5.0, 3.0, False, False),
        ("INT4 specificity", -10.0, -18.0, -2.0, True, False),
        ("DiD (INT4 $-$ FP16 spec.)", 8.0, 1.0, 16.0, True, False),
    ]
    enlarge = [
        ("FP16 identified K10", 7.0, 2.0, 12.0, True, False),
        ("FP16 random K10", 0.0, 0.0, 0.0, False, False),
        ("INT4 identified K10", 5.0, 1.0, 9.0, True, False),
        ("INT4 random K10", 3.0, 0.0, 7.0, False, False),
    ]
    l27 = [
        ("FP16 orig. n=50", -12.0, -22.0, -4.0, True, True),
        ("FP16 QA n=50", 0.0, 0.0, 0.0, False, False),
        ("INT4 orig. n=50", -8.0, -16.0, -2.0, True, True),
        ("INT4 QA n=50", 0.0, 0.0, 0.0, False, False),
        ("FP16 QA n=100", 0.0, 0.0, 0.0, False, False),
        ("INT4 QA n=100", 0.0, -3.0, 3.0, False, False),
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(COL_W, 6.55), sharex=False,
        gridspec_kw={"height_ratios": [1.35, 0.85, 1.15], "hspace": 0.42},
    )
    xlab = r"$\Delta$ follows_context (pp)"
    _forest_panel(
        axes[0], hminus, title=r"A  Gemma-2-9B  ·  $H^{-}$ vs random; DiD $+8$ $[+1,+16]$",
        xlabel=xlab, xlim=(-32, 20),
    )
    _forest_panel(
        axes[1], enlarge, title=r"B  Context K10 enlarge  ·  paired spec $+7$ vs $+2$; DiD $-5$ $[-9,-1]$",
        xlabel=xlab, xlim=(-6, 16),
    )
    axes[1].text(
        7.8, 0.5,
        "INT4 paired spec $+2$ $[+0,+5]$\n(null; DiD $-5$ $[-9,-1]$)",
        fontsize=6.1, color=MUTED, va="center",
    )
    _forest_panel(
        axes[2], l27, title="C  L27_H6  ·  INT4 n=100 is +0 [-3, +3], not a hard zero",
        xlabel=xlab, xlim=(-26, 8),
    )
    fig.subplots_adjust(left=0.32, right=0.98, top=0.97, bottom=0.06)
    _save(fig, OUT / "fig_gemma_audit.pdf")
    plt.close(fig)


def main() -> None:
    _apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_llama()
    fig_gemma()
    for name in ("fig_pipeline", "fig_llama_audit", "fig_gemma_audit"):
        print("wrote", OUT / f"{name}.pdf")
        print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
