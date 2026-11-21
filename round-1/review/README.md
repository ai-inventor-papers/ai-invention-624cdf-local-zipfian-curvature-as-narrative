# Review Report — Iteration 1 Audit

This directory contains the adversarial review of the internal research report for iteration 1 of the invention loop.

## Contents

- `.sdk_openhands_agent_struct_out.json` — Structured review output with scores, critiques, and assessment
- `.aii/manifest.yaml` — Artifact manifest for storage management

## Summary

**Overall Score: 3 (Reject)**

The research record documents a negative result on whether Zipf-Mandelbrot curvature drift correlates with semantic drift, but has three critical flaws:

1. **Dataset statistics in the report are factually incorrect** — the summary table reports function word ratio 0.568, type-token ratio 0.403, unique tokens 302.3, function word count 426.1. A random sample of 1,000 windows from the actual artifact (`full_data_out.json`) shows 0.548, 0.450, 337.2, 411.3 respectively. These are large systematic differences, not rounding errors.

2. **Per-section valid window counts are wrong** — report claims all 10 sections have 87 valid windows; the experiment artifact shows Section 3=85, Section 4=81, Section 6=83, Section 8=86, Section 9=85.

3. **Coverage is lost** — the user's original question was "How accurately does Zipf's law describe word frequencies in Pride and Prejudice?" This run tests a different hypothesis (curvature–semantic drift correlation) and never computes a global Zipf fit (exponent, R², goodness-of-fit) on the full novel.

The main experimental conclusions (correlations, baselines, negative result) are correctly traced to and recomputed from the experiment artifact.

## Artifacts Referenced

- `../gen_art/gen_art_dataset_1/` — Rolling windows dataset (KEEP)
- `../gen_art/gen_art_experiment_1/` — Experiment output (KEEP)

## Restoring Removed Files

No files were removed in this review step. All artifacts are marked `keep` in the manifest.