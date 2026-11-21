# Internal Research Report: Local Zipfian Curvature as Narrative Dynamics

This directory contains the internal research report for a two-iteration investigation of Zipf's law and the Zipf-Mandelbrot model applied to *Pride and Prejudice* (Project Gutenberg eBook #42671).

## Files

- `paper_draft.md` — The complete chronological research report (lab notebook style)
- `references.bib` — BibTeX bibliography with verified citations
- `figures/` — Rendered data figures (PDF and PNG)
  - `fig4.pdf/.png` — Global Zipf and Zipf-Mandelbrot fits on log-log scale
  - `fig5.pdf/.png` — Residuals by frequency rank bin
- `.aii/manifest.yaml` — Empty manifest (no large files to track)

## Iterations

### Iteration 1: Local Curvature–Semantic Drift Correlation
Tested whether local drift in Zipf-Mandelbrot curvature parameter correlates with semantic drift across rolling narrative windows. **Null result**: mean Pearson r = -0.025 ± 0.075, indistinguishable from shuffled baselines (p = 0.48 ± 0.23).

### Iteration 2: Global Zipfian Fit Accuracy (Addressing User's Original Question)
Computed global MLE fits for Zipf and Zipf-Mandelbrot models on the full novel. **Key finding**: Zipf-Mandelbrot (s=1.266, b=8.138) provides excellent fit (R²=0.967, KS=0.011), strongly outperforming standard Zipf (s=0.992, R²=0.741, KS=0.093) with ΔAIC = 21,449. The curvature parameter b is essential for modeling the elevated frequency of function words in English prose.

Also corrected dataset summary statistics from Iteration 1 (function word ratio 0.548, not 0.568; type-token ratio 0.450, not 0.403; unique tokens 337.4, not 302.3).

## Artifacts Referenced

- `art_e75eaOjXRd1x` — Dataset: Rolling Zipfian Narrative Windows
- `art_2pL5G-bOv532` — Experiment: Zipfian Curvature and Semantic Drift Analysis
- `art_dZwh_EvU-PqH` — Experiment: Global Zipf Fit on Pride and Prejudice
- `art_-8gUnATyh75D` — Evaluation: Global Zipfian Fit Accuracy Analysis
- `art_usMLbMsahxSw` — Evaluation: Global Zipfian Fit and Statistics Correction

All artifacts are stored in the pipeline workspace and referenced by ID in the report.