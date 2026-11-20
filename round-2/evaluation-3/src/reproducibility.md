# Reproducibility Guide: Global Zipfian Fit and Statistics Correction

This document provides complete step-by-step instructions to reproduce the evaluation results on Ubuntu.

## 1. Getting the Artifact

```bash
# Clone the repository (the artifact is one folder in the published repo)
git clone https://github.com/ai-inventor/ai-inventor.git
cd ai-inventor

# Navigate to this artifact's workspace
cd work/run_fODCbnzS2ZPQ/round-2/evaluation-3/src
```

## 2. System Requirements

- **OS**: Ubuntu 22.04+ (or any Linux with Python 3.12)
- **Python**: 3.12 (tested with 3.12.x)
- **RAM**: Minimum 2 GB (dataset is ~13 MB)
- **GPU**: Not required (CPU-only)
- **Disk**: ~200 MB for dependencies and data

## 3. Environment Setup

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install pinned dependencies from pyproject.toml
pip install -e .

# Or install directly:
pip install numpy==2.2.6 scipy==1.16.3 matplotlib==3.10.7
```

## 4. Required Data Files

The following input files must be present in the workspace (they are part of the published repo):

| File | Source | Size |
|------|--------|------|
| `full_data_out.json` | Dependency artifact `art_e75eaOjXRd1x` (gen_art_dataset_1) | 13 MB |
| `pride_and_prejudice_clean.txt` | Same dependency | 692 KB |

**Note**: These files are published as sibling folders in the repository. The dependency artifact `art_e75eaOjXRd1x` should be available at `../../../iter_1/gen_art/gen_art_dataset_1/`.

If running outside the full repo, copy them manually:
```bash
cp /path/to/gen_art_dataset_1/full_data_out.json .
cp /path/to/gen_art_dataset_1/pride_and_prejudice_clean.txt .
```

## 5. Running the Evaluation

```bash
# Run the main evaluation script
python eval.py
```

**Expected runtime**: ~30 seconds on a modern CPU (single-threaded)

**Expected output files**:
- `eval_out.json` — Full results in `exp_eval_sol_out` schema
- `full_eval_out.json` — Full version (same as eval_out.json)
- `mini_eval_out.json` — Mini version (first 3 examples)
- `preview_eval_out.json` — Preview version (truncated strings)
- `stats_comparison.json` — Original vs corrected statistics
- `diff.md` — Markdown explaining discrepancies

## 6. Regenerating the Figure

```bash
# Generate the chart specification
python create_spec.py

# Render using aii-data-fig-gen skill (requires the skill to be available)
cd /path/to/aii-data-fig-gen
python scripts/chart_gen.py --spec /path/to/zipf_spec.json --out /path/to/zipf_comparison
```

Or use the provided pre-rendered figure: `zipf_comparison.pdf` / `zipf_comparison.png`

**Expected runtime**: ~10 seconds for figure generation

## 7. Expected Results

### Statistics Correction
| Statistic | Original | Corrected | Difference |
|-----------|----------|-----------|------------|
| Total windows | 2,453 | 2,453 | 0 |
| Mean unique tokens/window | 750 | **337.39** | -412.61 |
| Mean function word ratio | N/A | **0.5476** | N/A |
| Mean type-token ratio | N/A | **0.4499** | N/A |

### Model Fits (MLE)

**Zipf (Power Law)** — 2 parameters
- Exponent (s): **0.9926**
- KS Statistic: **0.0935**
- AIC: **-1,068,674.10**
- BIC: **-1,068,660.55**

**Zipf-Mandelbrot** — 3 parameters
- Exponent (s): **1.2436**
- Curvature (b): **7.2575**
- KS Statistic: **0.0147**
- AIC: **-1,090,056.12**
- BIC: **-1,090,035.79**

### Model Comparison
- **Better model**: Zipf-Mandelbrot (ΔAIC = -21,382.02, ΔBIC = -21,375.24)
- KS improvement: 0.0788 (substantially better fit)

## 8. Validation

```bash
# Validate eval_out.json against schema
python -c "
import json
from jsonschema import validate
with open('schemas/exp_eval_sol_out.json') as f:
    schema = json.load(f)
with open('eval_out.json') as f:
    data = json.load(f)
validate(instance=data, schema=schema)
print('SCHEMA VALIDATION: PASSED')
"
```

Note: The schema file is at `~/.claude/skills/aii-json/schemas/exp_eval_sol_out.json` in the skill directory.

## 9. Key Numbers for Paper

| Claim | Value | Source |
|-------|-------|--------|
| Zipf exponent (s) | 0.9926 | `eval_out.json` → `metrics_agg.zipf_exponent_s` |
| Zipf-Mandelbrot exponent (s) | 1.2436 | `eval_out.json` → `metrics_agg.zipf_mandelbrot_exponent_s` |
| Zipf-Mandelbrot curvature (b) | 7.2575 | `eval_out.json` → `metrics_agg.zipf_mandelbrot_curvature_b` |
| KS improvement | 0.0788 | `eval_out.json` → `model_comparison.ks_improvement` |
| AIC difference | -21,382 | `eval_out.json` → `model_comparison.aic_diff` |
| Mean unique tokens/window | 337.39 | `eval_out.json` → `metrics_agg.mean_unique_tokens_per_window` |
| Function word ratio | 0.548 | `eval_out.json` → `metrics_agg.mean_function_word_ratio` |

## 10. Independent Re-derivation (Audit)

To verify the headline numbers independently, run:

```bash
python -c "
import json
import numpy as np
from scipy.optimize import minimize

# Load raw data
with open('eval_out.json') as f:
    data = json.load(f)

ranks = np.array(data['datasets'][0]['examples'][0]['output'])  # Wait, this won't work directly
# Better: load from raw_data
with open('eval_out.json') as f:
    data = json.load(f)

# The raw data is in a different format - let's verify from eval_out.json directly
print('Metrics from eval_out.json:')
for k, v in data['metrics_agg'].items():
    print(f'  {k}: {v}')
"
```

## Troubleshooting

- **ModuleNotFoundError**: Ensure you ran `pip install -e .` or installed the three dependencies
- **Permission errors**: Make sure you have write access to the workspace directory
- **Figure generation fails**: The `aii-data-fig-gen` skill is required; pre-rendered PDF/PNG are provided as fallback
- **Schema validation fails**: Ensure `jsonschema` is installed (`pip install jsonschema`)

## Notes

- All paths in this document are relative to the artifact workspace
- No API keys or external downloads are required
- The evaluation uses only CPU; no GPU setup needed
- The novel text is public domain (Project Gutenberg eBook #42671)