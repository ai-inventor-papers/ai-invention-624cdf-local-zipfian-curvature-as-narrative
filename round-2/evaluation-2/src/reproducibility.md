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
    numpy==2.2.6 \
    scipy==1.16.3 \
    loguru==0.7.3 \
    matplotlib==3.10.7 \
    jsonschema==4.25.1 \
    kiwisolver==1.4.9 \
    pillow==13.0.0 \
    contourpy==1.3.3 \
    cycler==0.12.1 \
    fonttools==4.63.0 \
    packaging==25.0 \
    pyparsing==3.2.5 \
    python-dateutil==2.9.0.post0
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
| `exp_eval_sol_out.json` | Full schema-compliant results (exp_eval_sol_out format) |
| `full_eval_out.json` | Copy of exp_eval_sol_out.json (full results) |
| `mini_eval_out.json` | First 3 examples only (for quick inspection) |
| `preview_eval_out.json` | Truncated strings (for quick inspection) |
| `fig_model_comparison.pdf/png` | Model comparison bar chart |
| `fig_residuals.pdf/png` | Residual-by-bin horizontal bar chart |
| `fig_rank_freq.pdf/png` | Rank-frequency scatter with model fits |

## 7. Key Results to Verify

The evaluation should produce these headline numbers (±0.001 tolerance for float values):

| Metric | Zipf (b=0) | Zipf-Mandelbrot |
|--------|------------|-----------------|
| Exponent (s) | 0.9925 | 1.2661 |
| Curvature (b) | 0.0 (fixed) | 8.1380 |
| KS Statistic | 0.0934 | 0.0112 |
| AIC | 1,581,276.42 | 1,559,827.09 |
| BIC | 1,581,286.15 | 1,559,846.54 |
| R² (log-log) | 0.7410 | 0.9671 |

**Model Comparison**:
- ΔAIC = 21,449.33 (Zipf-Mandelbrot preferred)
- ΔBIC = 21,439.61 (Zipf-Mandelbrot preferred)
- KS improves 8.4× (0.093 → 0.011)
- R² improves from 0.741 to 0.967

**Residual Analysis (mean residuals by bin)**:
| Bin | Zipf | Zipf-Mandelbrot |
|-----|------|-----------------|
| Ranks 1-648 (head) | 15.07 | 0.09 |
| Ranks 649-1296 | -0.73 | 1.54 |
| Ranks 1297-1944 | -1.90 | 0.40 |
| Ranks 1945-2592 | -2.06 | -0.04 |
| Ranks 2593-3240 | -1.96 | -0.18 |
| Ranks 3241-3888 | -1.83 | -0.25 |
| Ranks 3889-4536 | -2.06 | -0.64 |
| Ranks 4537-5184 | -1.81 | -0.51 |
| Ranks 5185-5832 | -1.48 | -0.29 |
| Ranks 5833-6479 (tail) | -1.22 | -0.12 |

## 8. Validation

Validate output schema:
```bash
python -c "
import json
from jsonschema import validate
with open('exp_eval_sol_out.json') as f: data = json.load(f)
with open('/Users/Shared/aii_data/.claude/skills/aii-json/schemas/exp_eval_sol_out.json') as f: schema = json.load(f)
validate(instance=data, schema=schema)
print('Schema validation: PASSED')
"
```

## 9. Figures Generation (Optional)

The figures are pre-generated but can be regenerated using aii-data-fig-gen:
```bash
# Requires aii-data-fig-gen skill environment
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec fig_model_comparison.json --out fig_model_comparison
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec fig_residuals.json --out fig_residuals
python /path/to/aii-data-fig-gen/scripts/chart_gen.py --spec fig_rank_freq.json --out fig_rank_freq
```

## 10. Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies from step 3 are installed
- **Font warnings (CMU Serif)**: Harmless; matplotlib falls back to DejaVu Serif
- **File not found for input text**: Ensure `../gen_art_dataset_1/pride_and_prejudice_clean.txt` exists relative to this directory, or update the path in `eval.py`
- **Schema validation fails**: Check that `exp_eval_sol_out.json` matches the exp_eval_sol_out schema structure

## 11. Notes

- The evaluation uses MLE (Maximum Likelihood Estimation) via SciPy's Nelder-Mead (ZM) and bounded scalar minimization (Zipf)
- No random seeds are involved; results are deterministic
- The tokenization uses regex `[a-zA-Z]+` on lowercase text
- All numerical results in the paper/figures should match the values in `exp_eval_sol_out.json`
- This artifact follows computational linguistics best practices (Clauset et al., 2009; Corral et al., 2015)