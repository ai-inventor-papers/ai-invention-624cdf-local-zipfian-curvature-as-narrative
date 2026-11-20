# Reproducibility Instructions

## 1. Getting This Artifact

This artifact is published as part of a public GitHub repository. To reproduce:

```bash
git clone <repository-url>
cd <repository-root>/round-2/evaluation-2/src
```

## 2. System Requirements

- **OS**: Ubuntu 22.04+ (or macOS with Homebrew)
- **Python**: 3.12+
- **System packages**: `build-essential`, `python3.12-venv` (Ubuntu)
- **Hardware**: CPU only (no GPU required), ~1 GB RAM, ~100 MB disk space

## 3. Environment Setup

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install pinned dependencies (matching pyproject.toml)
pip install \
    attrs==26.1.0 \
    contourpy==1.4.0 \
    cycler==0.12.1 \
    fonttools==4.68.0 \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    kiwisolver==1.5.1 \
    loguru==0.8.0 \
    matplotlib==3.11.2 \
    numpy==2.5.5 \
    packaging==26.3 \
    pillow==13.0.0 \
    pyparsing==3.3.3 \
    python-dateutil==4.0.0 \
    referencing==0.37.0 \
    rpds-py==2026.9.1 \
    scipy==1.18.1 \
    typing-extensions==4.16.0 \
    tzdata==2026.6
```

## 4. Required Input Data

This evaluation requires the clean text of Pride and Prejudice from the dataset artifact `art_e75eaOjXRd1x` (published as a sibling folder in the repository):

```
../gen_art_dataset_1/pride_and_prejudice_clean.txt
```

If running outside the published repository structure, download the clean text from Project Gutenberg (eBook #42671) and remove the header/footer boilerplate.

## 5. Running the Evaluation

```bash
# Ensure you're in the artifact directory
cd round-2/evaluation-2/src

# Activate environment
source .venv/bin/activate

# Run evaluation (takes ~2 seconds on modern CPU)
python eval.py
```

**Approximate runtime**: 2-5 seconds on a modern CPU (single-threaded)

## 6. Expected Output Files

After successful execution, you should get:

| File | Description |
|------|-------------|
| `eval_out.json` | Full schema-compliant results (exp_eval_sol_out format) |
| `full_eval_out.json` | Copy of eval_out.json (full results) |
| `mini_eval_out.json` | First 3 examples only (for quick inspection) |
| `preview_eval_out.json` | Truncated strings (for quick inspection) |
| `figures/fig1_loglog.pdf/png` | Rank-frequency distribution (log-log) |
| `figures/fig2_residuals.pdf/png` | Residuals vs fitted values |
| `figures/fig3_model_comparison.pdf/png` | Model comparison bar chart |
| `figures/fig4_residual_bins.pdf/png` | Residual analysis by rank bin |
| `figures/fig5_panel.pdf/png` | Multi-panel composite figure |
| `logs/run.log` | Execution log with timestamps |

## 7. Key Results to Verify

The evaluation should produce these headline numbers (±0.001 tolerance for float values):

| Metric | Zipf | Zipf-Mandelbrot |
|--------|------|-----------------|
| Exponent (s) | 1.0010 | 1.2710 |
| Curvature (b) | 0.0 (fixed) | 8.2490 |
| Log-Likelihood | -783,709.46 | -772,899.63 |
| KS Statistic | 0.1012 | 0.0114 |
| AIC | 1,567,420.92 | 1,545,803.26 |
| BIC | 1,567,427.66 | 1,545,816.74 |
| R² (log-log) | 0.7560 | 0.9659 |

**Model Comparison**:
- ΔAIC = -21,617.66 (Zipf-Mandelbrot preferred)
- ΔBIC = -21,610.92 (Zipf-Mandelbrot preferred)

**Residual Analysis (mean residuals by bin)**:
| Bin | Zipf | Zipf-Mandelbrot |
|-----|------|-----------------|
| Head (top 10%) | 14.21 | 0.092 |
| Mid (middle 80%) | -1.63 | 0.007 |
| Tail (bottom 10%) | -1.20 | -0.147 |

## 8. Validation

Validate output schema:
```bash
python -c "
import json
from jsonschema import validate
with open('eval_out.json') as f: data = json.load(f)
with open('/Users/Shared/aii_data/.claude/skills/aii-json/schemas/exp_eval_sol_out.json') as f: schema = json.load(f)
validate(instance=data, schema=schema)
print('Schema validation: PASSED')
"
```

## 9. Figures Generation (Optional)

The figures are pre-generated but can be regenerated:
```bash
cd figures
# Requires aii-data-fig-gen skill environment
python -c "
import json, sys
sys.path.insert(0, '/path/to/aii-data-fig-gen/scripts')
from chart_gen import main
import subprocess
for spec in ['fig1_loglog.json', 'fig2_residuals.json', 'fig3_model_comparison.json', 'fig4_residual_bins.json', 'fig5_panel.json']:
    subprocess.run(['python', '/path/to/aii-data-fig-gen/scripts/chart_gen.py', '--spec', spec, '--out', spec.replace('.json','')])
"
```

## 10. Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies from step 3 are installed
- **Font warnings (CMU Serif)**: Harmless; matplotlib falls back to DejaVu Serif
- **File not found for input text**: Ensure `../gen_art_dataset_1/pride_and_prejudice_clean.txt` exists relative to this directory, or update the path in `eval.py`
- **Schema validation fails**: Check that `eval_out.json` matches the exp_eval_sol_out schema structure

## 11. Notes

- The evaluation uses MLE (Maximum Likelihood Estimation) via SciPy's Nelder-Mead (ZM) and bounded scalar minimization (Zipf)
- No random seeds are involved; results are deterministic
- The tokenization uses regex `[a-zA-Z]+` on lowercase text
- All numerical results in the paper/figures should match the values in `eval_out.json`