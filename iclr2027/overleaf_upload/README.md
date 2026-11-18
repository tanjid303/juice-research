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

Figure 1 is the author-supplied vector `pipeline_overview.pdf` (project root, next to `main.tex`). Do not regenerate it. Remaining PDFs in `figures/` are generated from locked result JSONs:

```bash
.venv/Scripts/python.exe iclr2027/figures/generate_figures.py
```

| File | What it shows |
| --- | --- |
| `pipeline_overview.pdf` | Figure 1: Identify → dual-run steer → GPTQ INT4 transfer |
| `figures/fig1_juice_dualrun.pdf` | Dual-run JuICE schematic |
| `figures/fig2_delta_follows_context.pdf` | Llama $\Delta$ follows_context (context K10 vs parametric repairs) |
| `figures/fig3_directional_blindspot.pdf` | Llama ID scatter: L1_H4 vs directional suppressors |
| `figures/fig4_n64_vs_n128.pdf` | GPTQ calib-size robustness |
| `figures/fig5_gemma_deltas.pdf` | Gemma enlarge + suppress + DiD |
| `figures/fig6_cross_family.pdf` | Headline: Llama transfers context K10; Gemma transfers L27_H6 |
| `figures/fig7_gemma_blindspot.pdf` | Gemma ID scatter: L27_H6 vs L31_H7 / L39_H6 |

## Overleaf

Upload the contents of `overleaf_upload/` (main, bib, style files, `figures/*.pdf`).
