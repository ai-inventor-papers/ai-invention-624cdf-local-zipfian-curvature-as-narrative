# Global Zipfian Fit and Statistics Correction

This artifact performs a systematic correction of dataset summary statistics for *Pride and Prejudice* and computes global Zipfian fit accuracy using Maximum Likelihood Estimation (MLE) and model comparison.

## Summary

- **Dataset**: *Pride and Prejudice* (Project Gutenberg eBook #42671), ~123K tokens
- **Task**: Fit Zipf (Power Law) and Zipf-Mandelbrot models to the word frequency distribution using MLE (not log-log OLS)
- **Comparison**: AIC/BIC and KS statistic to determine if Zipf-Mandelbrot's additional curvature parameter provides meaningful improvement

## Key Results

### Statistics Correction
| Statistic | Original Claim | Corrected Value | Difference |
|-----------|---------------|-----------------|------------|
| Total windows | 2,453 | 2,453 | 0 |
| Mean unique tokens/window | 750 | 337.39 | -412.61 |
| Mean function word count | N/A | 410.73 | N/A |
| Mean function word ratio | N/A | 0.548 | N/A |
| Mean type-token ratio | N/A | 0.450 | N/A |

**Key finding**: The original claim of "mean unique tokens per window: 750" was incorrect — the window size is 750 tokens, but unique tokens per window is ~337.

### Model Fits

**Zipf (Power Law) — 2 parameters**
- Exponent (s): 0.9926
- Normalization (C): 12,806.79
- Log-likelihood: 534,339.05
- AIC: -1,068,674.10
- BIC: -1,068,660.55
- KS Statistic: 0.0935

**Zipf-Mandelbrot — 3 parameters**
- Exponent (s): 1.2436
- Curvature (b): 7.2575
- Normalization (C): 61,278.09
- Log-likelihood: 545,031.06
- AIC: -1,090,056.12
- BIC: -1,090,035.79
- KS Statistic: 0.0147

### Model Comparison
- **Better model**: Zipf-Mandelbrot (lower AIC/BIC)
- AIC difference: -21,382.02 (strong evidence for Zipf-Mandelbrot)
- BIC difference: -21,375.24 (very strong evidence for Zipf-Mandelbrot)
- KS improvement: 0.0788 (Zipf-Mandelbrot fits much better)

The Zipf-Mandelbrot model provides a substantially better fit to the empirical frequency distribution, confirming that the 2-parameter model better captures deviations in the head (function words) and tail (hapax legomena) of the distribution.

## Files

| File | Description |
|------|-------------|
| `eval.py` | Main evaluation script — computes statistics, fits models, generates outputs |
| `create_spec.py` | Generates chart specification from evaluation results |
| `eval_out.json` | Full evaluation results in `exp_eval_sol_out` schema format |
| `stats_comparison.json` | Original vs. corrected statistics comparison |
| `diff.md` | Markdown document explaining discrepancies |
| `zipf_comparison.pdf` | Log-log plot: empirical vs. fitted Zipf and Zipf-Mandelbrot |
| `zipf_comparison.png` | PNG version of the figure |
| `zipf_spec.json` | Chart specification for figure generation |
| `full_data_out.json` | Windowed dataset from dependency (13 MB) |
| `full_method_out.json` | Prior experiment results from dependency (424 KB) |
| `pride_and_prejudice_clean.txt` | Clean novel text (692 KB) |
| `preview_data_out.json` | Preview of windowed dataset |
| `preview_method_out.json` | Preview of prior experiment |

## How to Run

```bash
# Run the evaluation
uv run eval.py

# Regenerate the figure
uv run create_spec.py
cd /Users/Shared/aii_data/.claude/skills/aii-data-fig-gen
python scripts/chart_gen.py --spec /path/to/zipf_spec.json --out /path/to/zipf_comparison
```

## Restoring Removed Files

The following files can be regenerated or re-downloaded:

- `full_data_out.json`: `cp /Volumes/aii-s20007/work/run_fODCbnzS2ZPQ/round-1/dataset-1/src/full_data_out.json .`
- `full_method_out.json`: `cp /Volumes/aii-s20007/work/run_fODCbnzS2ZPQ/round-1/experiment-1/src/full_method_out.json .`
- `pride_and_prejudice_clean.txt`: `cp /Volumes/aii-s20007/work/run_fODCbnzS2ZPQ/round-1/dataset-1/src/pride_and_prejudice_clean.txt .`
- `zipf_comparison.png`: Run `uv run create_spec.py` then use `aii-data-fig-gen` skill
- `zipf_spec.json`: Run `uv run create_spec.py`
- `preview_data_out.json`: `cp /Volumes/aii-s20007/work/run_fODCbnzS2ZPQ/round-1/dataset-1/src/preview_data_out.json .`
- `preview_method_out.json`: `cp /Volumes/aii-s20007/work/run_fODCbnzS2ZPQ/round-1/experiment-1/src/preview_method_out.json .`
- `test_load.py`: Simple test script (can be recreated)

## Methodology Notes

- **MLE fitting**: Used Poisson likelihood for rank-frequency data, avoiding the well-known bias of log-log OLS regression
- **Zipf-Mandelbrot form**: f(r) = C × (r + b)^(-s) where r is rank, s is exponent, b is curvature
- **Model selection**: AIC/BIC account for the extra parameter in Zipf-Mandelbrot; both strongly favor it
- **KS statistic**: Computed on cumulative distribution functions (model vs empirical)
- **All computations**: Single-threaded, < 1 minute runtime, no GPU required