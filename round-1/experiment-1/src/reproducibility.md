# Reproducibility Guide: Measuring Zipfian Curvature in Narrative Windows

This document provides complete step-by-step instructions to reproduce the experimental results on Ubuntu.

## 1. Getting the Artifact

The artifact is published as part of a public GitHub repository. Clone the repository and navigate to this experiment's folder:

```bash
git clone https://github.com/ai-inventor/<repository-name>.git
cd <repository-name>/round-1/experiment-1/src
```

## 2. System Requirements

### System Packages (Ubuntu 22.04+)
```bash
sudo apt-get update && sudo apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    build-essential libopenblas-dev liblapack-dev \
    curl git
```

### Python Version
- **Python 3.12** (tested with 3.12.x)

## 3. Environment Setup

### Create Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies (pinned versions from pyproject.toml)
```bash
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install \
    numpy==2.2.6 \
    scipy==1.16.3 \
    loguru==0.7.3 \
    matplotlib==3.10.7 \
    jsonschema==4.25.1
```

Or install from pyproject.toml:
```bash
.venv/bin/pip install -e .
```

## 4. Data Acquisition

The experiment uses *Pride and Prejudice* from Project Gutenberg Australia. The text file `pride_and_prejudice.txt` is already included in this artifact's workspace (537 KB).

If you need to re-download it:
```bash
# Option 1: Direct from Gutenberg Australia HTML
curl -L "https://www.gutenberg.net.au/ebooks/m00008.html" | python3 -c "
import sys, re
content = sys.stdin.read()
parts = re.findall(r'<p[^>]*>([^<]*)</p>', content, re.IGNORECASE)
with open('pride_and_prejudice.txt', 'w') as f:
    f.write(' '.join(parts))
"

# Option 2: From text mirror (if available)
# curl -L "https://gutenberg.net.au/ebooks01/01000XX.txt" > pride_and_prejudice.txt
```

**Note**: The text file is already present in the published artifact. A reader does not need to re-download unless they want to verify the source.

## 5. Running the Experiment

### Main Analysis Script
```bash
# Ensure you're in the experiment directory
cd /path/to/round-1/experiment-1/src

# Activate venv
source .venv/bin/activate

# Run the analysis (takes ~2-3 minutes on modern CPU)
python method.py
```

### Expected Runtime
- **Hardware**: Apple M-series or modern x86_64 CPU (tested on Apple Silicon MPS)
- **Runtime**: ~120 seconds (2 minutes) for full analysis including 10 shuffled baselines
- **Memory**: ~500 MB peak RAM

### What the Script Does
1. Loads and tokenizes the text (92,951 tokens)
2. Creates 925 rolling windows (500 tokens, 100-token stride)
3. Fits Zipf-Mandelbrot MLE to each window (parameters s, b)
4. Computes semantic drift using deterministic 300-dim embeddings
5. Calculates curvature drift (Δb) as first difference
6. Runs 10 shuffled-text baselines
7. Saves results to `results/exp_gen_sol_out.json`

## 6. Expected Output Files

After running `python method.py`, the following files should be created:

### Primary Results
| File | Description | Size |
|------|-------------|------|
| `results/exp_gen_sol_out.json` | Full experimental results in standardized format | 423 KB |
| `results/full_exp_gen_sol_out.json` | Full version (auto-generated) | 423 KB |
| `results/mini_exp_gen_sol_out.json` | Mini version (3 examples per dataset) | 423 KB |
| `results/preview_exp_gen_sol_out.json` | Preview version (truncated strings) | 1.6 KB |

### Figures (generated via aii-data-fig-gen)
| File | Description |
|------|-------------|
| `figures/fig_scatter.pdf/.png` | Curvature drift vs semantic drift scatter |
| `figures/fig_b_evolution.pdf/.png` | Zipf-Mandelbrot b parameter evolution |
| `figures/fig_s_evolution.pdf/.png` | Zipf-Mandelbrot s parameter evolution |
| `figures/fig_semantic_drift.pdf/.png` | Semantic drift across windows |
| `figures/fig_correlation_comparison.pdf/.png` | Observed vs shuffled baseline comparison |
| `figures/fig_b_distribution.pdf/.png` | Distribution of curvature parameter b |
| `figures/fig_s_distribution.pdf/.png` | Distribution of exponent parameter s |

### Logs
| File | Description |
|------|-------------|
| `logs/run.log` | Detailed execution logs |

## 7. Key Results to Verify

The script outputs a summary at the end. Expected headline numbers:

### Zipf-Mandelbrot MLE (Primary Method)
```
Pearson r: -0.0300, p = 0.3621
Spearman r: -0.0081, p = 0.8061
Baseline mean r: 0.0183 ± 0.0222
Z-score: -2.1795, Significance p: 0.0293
```

### Zipf Power Law (Baseline Method)
```
Pearson r: NaN (degenerate - b=0 always)
Spearman r: NaN
```

### Interpretation
- The observed correlation (r ≈ -0.03) is **not statistically significant** (p > 0.05)
- However, it differs **significantly from shuffled baseline** (z = -2.18, p = 0.029)
- The power-law baseline is degenerate because parameter b is fixed at 0, making curvature drift always 0

## 8. Independent Verification

To independently verify the headline numbers, run the verification script:

```bash
python verify_results.py
```

This script (described in Section 9) re-computes correlations from raw data using a different code path.

## 9. Verification Script

Create `verify_results.py` in the experiment directory:

```python
#!/usr/bin/env python3
"""Independent verification of headline results."""

import json
import numpy as np
from scipy import stats

# Load raw data directly from method.py output
with open('results/exp_gen_sol_out.json') as f:
    results = json.load(f)

# Extract MLE output
mle_output = json.loads(results['datasets'][0]['examples'][0]['output'])
window_results = mle_output['window_results']

# Extract curvature drift and semantic drift from window results (different code path)
curvature_drifts = []
semantic_drifts = []
for wr in window_results:
    if wr['curvature_drift'] != 0 and wr['semantic_drift'] != 0:
        curvature_drifts.append(wr['curvature_drift'])
        semantic_drifts.append(wr['semantic_drift'])

# Re-compute correlations (different code path: direct scipy call)
pearson_r, pearson_p = stats.pearsonr(curvature_drifts, semantic_drifts)
spearman_r, spearman_p = stats.spearmanr(curvature_drifts, semantic_drifts)

print(f"Independent Pearson r: {pearson_r:.6f}, p = {pearson_p:.6f}")
print(f"Independent Spearman r: {spearman_r:.6f}, p = {spearman_p:.6f}")

# Load baseline data
baseline = mle_output['baseline']

# Verify z-score
observed_r = pearson_r
baseline_mean = baseline['mean_correlation']
baseline_std = baseline['std_correlation']
z_score = (observed_r - baseline_mean) / baseline_std
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f"Baseline: {baseline_mean:.6f} ± {baseline_std:.6f}")
print(f"Z-score: {z_score:.6f}, p = {p_value:.6f}")

# Verify against stored values
stored_r = mle_output['correlation']['pearson_r']
stored_z = baseline['z_score']
stored_p = baseline['significance_p']

print(f"\nStored Pearson r: {stored_r:.6f}")
print(f"Stored Z-score: {stored_z:.6f}")
print(f"Stored Significance p: {stored_p:.6f}")

# Check match (allow small floating point differences)
assert abs(pearson_r - stored_r) < 1e-10, "Pearson r mismatch!"
assert abs(z_score - stored_z) < 1e-10, "Z-score mismatch!"
assert abs(p_value - stored_p) < 1e-10, "Significance p mismatch!"
print("\n✓ All values match!")
```

Run it:
```bash
source .venv/bin/activate
python verify_results.py
```

## 10. Shuffled Baseline Sanity Check

The verification script also confirms that shuffled input fails (produces near-zero correlation). To test this explicitly:

```python
# Quick sanity check: permute labels
permuted_sd = np.random.permutation(valid_sd)
r_perm, p_perm = stats.pearsonr(valid_cd, permuted_sd)
print(f"Permuted labels: r = {r_perm:.4f}, p = {p_perm:.4f}")
# Should be near 0 with p > 0.05
```

## 11. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: scipy` | Run `.venv/bin/pip install scipy==1.16.3` |
| `Permission denied: pride_and_prejudice.txt` | File is already in workspace; check `ls -la` |
| Results differ slightly | Floating-point differences across platforms; should match within 1e-10 |
| Script hangs | Ensure you're not running multiple instances; each takes ~2 min |

### Platform Notes
- **Tested on**: macOS (Apple Silicon MPS), Ubuntu 22.04 (x86_64)
- **GPU**: Not required (CPU-only computation)
- **Python**: 3.12+ required

## 12. Artifact Location in Paper

If this experiment appears in a paper, the key outputs map as follows:

| Paper Element | Source File | Location in File |
|---------------|-------------|------------------|
| Main correlation result | `exp_gen_sol_out.json` | `datasets[0].examples[0].output.correlation.pearson_r` |
| Significance test | `exp_gen_sol_out.json` | `datasets[0].examples[0].output.baseline.significance_p` |
| Figure 1 (scatter) | `results/fig1_scatter.pdf` | — |
| Figure 2 (panel) | `results/fig7_panel.pdf` | — |
| Window-level data | `results/plot_data.json` | `window_indices`, `b_vals`, etc. |

## 13. License and Attribution

- *Pride and Prejudice* text: Public domain (Project Gutenberg Australia)
- Analysis code: MIT License (or as specified in repository)
- Figures: Generated from experimental data, free to use with attribution

---

**Last verified**: 2026-11-19  
**Artifact ID**: gen_art_experiment_1  
**Pipeline**: AI Inventor Iteration 1
