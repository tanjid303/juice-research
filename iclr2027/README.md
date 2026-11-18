# Whose Word Survives Compression?

ICLR 2027 submission bundle. All numbers are locked to `writeup/stage1_stage2_consolidated.md` and the JSON sources listed there. Do not invent cells (no Gemma Stage 1, no Gemma native INT4 re-ID, no Gemma reverse transfer).

## Compile

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Figures

PDFs in `figures/` are generated from locked result JSONs:

```bash
.venv/Scripts/python.exe iclr2027/figures/generate_figures.py
```

| File | What it shows |
| --- | --- |
| `fig0_overview.pdf` | Intro pipeline: Identify → Steer → Transfer, then Llama / Gemma outcomes |
| `fig1_juice_dualrun.pdf` | Dual-run JuICE schematic |
| `fig2_delta_follows_context.pdf` | Llama $\Delta$ follows_context (context K10 vs parametric repairs) |
| `fig3_directional_blindspot.pdf` | Llama ID scatter: L1_H4 vs directional suppressors |
| `fig4_n64_vs_n128.pdf` | GPTQ calib-size robustness |
| `fig5_gemma_deltas.pdf` | Gemma enlarge + suppress + DiD |
| `fig6_cross_family.pdf` | Headline: Llama transfers context K10; Gemma transfers L27_H6 |
| `fig7_gemma_blindspot.pdf` | Gemma ID scatter: L27_H6 vs L31_H7 / L39_H6 |

## Overleaf

Upload the contents of `overleaf_upload/` (main, bib, style files, `figures/*.pdf`).
