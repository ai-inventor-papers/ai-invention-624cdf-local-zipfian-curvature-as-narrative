# Reproducibility Guide: Global Zipf Fit on Pride and Prejudice

This document provides complete step-by-step instructions to reproduce the exact results from this experiment on Ubuntu.

## 1. Get This Artifact

```bash
# Clone the repository (replace with actual repo URL when published)
git clone <repository-url>
cd <repository-root>/round-2/experiment-2/src
```

## 2. System Requirements

- **OS**: Ubuntu 22.04+ (or compatible Linux)
- **Python**: 3.12+
- **RAM**: 4 GB minimum (dataset is ~13 MB)
- **Disk**: 500 MB for dependencies and outputs
- **GPU**: Not required (CPU-only experiment)

## 3. Environment Setup

### 3.1 Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install Dependencies (pinned versions)

```bash
pip install --upgrade pip
pip install -r pyproject.toml
```

Or manually with exact versions from `pyproject.toml`:
```bash
pip install \
    numpy==2.2.6 \
    scipy==1.16.3 \
    matplotlib==3.10.7 \
    pandas==2.3.3 \
    scikit-learn==1.7.2 \
    loguru==0.7.3 \
    jsonschema==4.25.1 \
    PyYAML==6.0.3 \
    tqdm==4.67.1
```

### 3.3 Verify Installation

```bash
python -c "import numpy, scipy, matplotlib, pandas, sklearn, loguru, jsonschema, yaml, tqdm; print('All packages loaded successfully')"
```

## 4. Required Data

This experiment depends on the dataset from iteration 1: `art_e75eaOjXRd1x` (Rolling Zipfian Narrative Windows from Pride and Prejudice).

### 4.1 Automatic (if sibling artifact exists)

If the repository publishes sibling artifacts, the data will be at:
```
../gen_art_dataset_1/full_data_out.json
```

### 4.2 Manual Download (if needed)

If running independently, you need the full dataset from the dependency artifact. The dataset is NOT included in this artifact's workspace.

**To obtain the data:**
1. The source is Project Gutenberg eBook #42671 (Pride and Prejudice)
2. Or run the data generation script from the dependency artifact:
   ```bash
   cd ../gen_art_dataset_1 && python data.py
   ```

**Note**: The user upload folder is private and not published. A reader must supply their own copy of `full_data_out.json` at `../gen_art_dataset_1/full_data_out.json` relative to this workspace.

## 5. Running the Experiment

### 5.1 Main Experiment (MLE fitting + bootstrap CIs)

```bash
# Run from the experiment directory
cd /path/to/gen_art_experiment_2
uv run method.py
```

**Expected runtime**: ~30-60 seconds on a modern CPU (bootstrap CIs with 100 iterations)

**Expected output files**:
- `exp_gen_sol_out.json` - Primary output (schema-validated)
- `results.json` - Raw results for easy access
- `full_exp_gen_sol_out.json`, `mini_exp_gen_sol_out.json`, `preview_exp_gen_sol_out.json` - Sized variants
- `full_results.json`, `preview_results.json` - Sized raw results

### 5.2 Generate Figures

```bash
uv run generate_figures.py
```

**Expected output in `figures/`**:
- `fig1_rank_freq.pdf/png` - Log-log rank-frequency plot
- `fig2_residuals.pdf/png` - Residuals by rank region
- `fig3_gof.pdf/png` - Goodness-of-fit comparison
- `fig4_bootstrap_ci.pdf/png` - Bootstrap confidence intervals
- `*_spec.json` - Figure specifications (regenerable)

### 5.3 Validate Output Schema

```bash
python aii_json_format_mini_preview.py --format exp_gen_sol_out --input exp_gen_sol_out.json
```

## 6. Expected Results (Headline Numbers)

After running `method.py`, you should get these exact values in `results.json` and `exp_gen_sol_out.json`:

### Model Parameters
- **Zipf s**: 0.9924272695949843
- **Zipf-Mandelbrot s**: 1.2671043263376347
- **Zipf-Mandelbrot b**: 8.164884585229958

### Goodness-of-Fit
- **Zipf KS statistic**: 0.09349229078785598
- **Zipf-Mandelbrot KS statistic**: 0.011195099331401032
- **Zipf R² (log-log)**: 0.7390240538164308
- **Zipf-Mandelbrot R² (log-log)**: 0.9664875488563884
- **Zipf AIC**: 1577534.8611097834
- **Zipf-Mandelbrot AIC**: 1556072.8874704114
- **ΔAIC (Zipf - Zipf-Mandelbrot)**: 21461.973639372038

### Bootstrap 95% Confidence Intervals (n=100, seed=42)
- **Zipf s**: [0.9911004911652695, 0.9952106434765149]
- **Zipf-Mandelbrot s**: [1.258252762876514, 1.2690043084742577]
- **Zipf-Mandelbrot b**: [7.943007670724084, 8.41004930743093]

### Novel Statistics
- **Total tokens**: 123,350
- **Vocabulary size**: 6,425
- **Max frequency**: 4,338 ("the")
- **Min frequency**: 1 (hapax legomena)

### Residual Analysis (Zipf-Mandelbrot)
- **Head (top 10%)**: mean residual = 0.074, mean relative = 0.054
- **Body (10-50%)**: mean residual = 0.445, mean relative = 0.027
- **Tail (bottom 50%)**: mean residual = -0.371, mean relative = -0.230

## 7. Verification Steps

### 7.1 Check JSON Schema Validation Passes
```bash
python -c "
import json
from jsonschema import Draft7Validator
with open('schemas/exp_gen_sol_out.json') as f:
    schema = json.load(f)
with open('exp_gen_sol_out.json') as f:
    data = json.load(f)
errors = list(Draft7Validator(schema).iter_errors(data))
print('Validation:', 'PASSED' if not errors else 'FAILED')
"
```

### 7.2 Check Figure Files Exist
```bash
ls -la figures/*.pdf figures/*.png
```

### 7.3 Spot-Check Key Numbers
```bash
python -c "
import json
with open('results.json') as f:
    r = json.load(f)
print('s_zipf:', r['zipf_params']['s'])
print('s_zm:', r['zipf_mandelbrot_params']['s'])
print('b_zm:', r['zipf_mandelbrot_params']['b'])
print('KS_zm:', r['ks_zm'])
print('R2_zm:', r['r2_zm'])
print('AIC_diff:', r['aic_diff'])
"
```

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure venv is activated and dependencies installed |
| `FileNotFoundError: full_data_out.json` | Ensure dependency artifact data is at `../gen_art_dataset_1/full_data_out.json` |
| `Optimization failed` | L-BFGS-B may need different initial guesses; try modifying bounds in `method.py` |
| Figures fail to render | Ensure `matplotlib` has a display or use `Agg` backend (already default) |
| Schema validation fails | Check that `exp_gen_sol_out.json` matches the required structure |

## 9. Environment Variables

No API keys or special environment variables required for this experiment.

## 10. Hardware Used for Original Run

- **Machine**: macOS Apple Silicon (M-series)
- **CPU**: 6 vCPUs
- **RAM**: 24 GB (20 GB allocated)
- **GPU**: Not used (CPU-only)
- **Runtime**: ~45 seconds for `method.py`, ~10 seconds for `generate_figures.py`

## 11. Output Files Reference

| File | Description | In Paper |
|------|-------------|----------|
| `exp_gen_sol_out.json` | Primary schema-validated output | Main results table |
| `results.json` | Raw numeric results | Supplementary |
| `figures/fig1_rank_freq.pdf` | Log-log rank-frequency plot | Figure 1 |
| `figures/fig2_residuals.pdf` | Residuals by region | Figure 2 |
| `figures/fig3_gof.pdf` | Goodness-of-fit comparison | Figure 3 |
| `figures/fig4_bootstrap_ci.pdf` | Parameter confidence intervals | Figure 4 |

---

**Note**: All paths in this document are relative to the workspace root (`gen_art_experiment_2/`). The published repository will have this folder as a sibling to `gen_art_dataset_1/`.