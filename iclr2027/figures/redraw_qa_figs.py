"""Redraw fig2, fig5, fig6 only from question-aware numbers."""
from generate_figures import _apply_style, fig2_deltas, fig5_gemma_deltas, fig6_cross_family, OUT

if __name__ == "__main__":
    _apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig2_deltas()
    fig5_gemma_deltas()
    fig6_cross_family()
    for name in (
        "fig2_delta_follows_context",
        "fig5_gemma_deltas",
        "fig6_cross_family",
    ):
        print("wrote", OUT / f"{name}.pdf")
        print("wrote", OUT / f"{name}.png")
