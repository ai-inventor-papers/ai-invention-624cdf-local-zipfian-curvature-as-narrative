# Internal Research Report: Local Zipfian Curvature as Narrative Dynamics

This directory contains the internal research report for Iteration 1 of the AI Inventor run investigating whether local drift in Zipf-Mandelbrot curvature correlates with semantic drift in narrative text.

## Contents

- `paper_draft.md` — The main research report (chronological lab notebook format)
- `figures/` — Rendered data figures (PDF and PNG)
  - `fig1_v0.pdf/.png` — Per-section Pearson correlations vs shuffled baseline
  - `fig2_v0.pdf/.png` — Zipf-Mandelbrot parameter means across sections
  - `fig3_v0.pdf/.png` — Z-scores of observed correlations against shuffled baseline
- `style_exemplars.md` — Curated passages from 5 field papers for style reference
- `domain_terms.json` — 56 field vocabulary terms with glosses and sources
- `field_extras.json` — 5 extra deliverables identified from exemplar papers
- `references.bib` — BibTeX bibliography (7 entries fetched from Semantic Scholar)
- `references.json` — Fetch record for bibliography entries
- `.aii/manifest.yaml` — Artifact retention decisions (empty — all files under auto-keep floor)

## Report Summary

**Hypothesis**: Local drift in the curvature (b parameter) of the Zipf-Mandelbrot frequency distribution in a novel is a reliable proxy for narrative semantic intensity (rate of new concept introduction).

**Method**: Partitioned *Pride and Prejudice* (~123K tokens) into rolling windows, fit Zipf-Mandelbrot via MLE, computed semantic drift via cosine distance between window centroids (deterministic 300-dim embeddings), correlated curvature drift (Δb) with semantic drift against 10 shuffled-text baselines per section.

**Results**: No significant correlation found. Mean Pearson r = -0.025 ± 0.075 across 10 sections, indistinguishable from shuffled baseline (p = 0.48 ± 0.23). The Zipf-Mandelbrot model fits well (mean R² = 0.967) but curvature drift does not track semantic drift. Standard Zipf power law is degenerate for this analysis (b ≡ 0).

**Key artifacts**: Dataset (2,453 windows of 750 tokens) and experiment (10 sections × 2 methods) with full per-window and per-section results.

## Figures

All figures are data charts rendered deterministically from the experiment's own numbers using the `aii-data-fig-gen` skill. They can be regenerated from the spec files in `figures/` if needed.

```bash
# Regenerate figures
python /Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/fig1_spec.json --out figures/fig1_v0
python /Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/fig2_spec.json --out figures/fig2_v0
python /Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec figures/fig3_spec.json --out figures/fig3_v0
```