# Global Zipfian Fit Accuracy Analysis

This artifact evaluates how accurately Zipf's law and the Zipf-Mandelbrot distribution describe word frequencies in Pride and Prejudice (Project Gutenberg eBook #42671).

## Methodology

Following computational linguistics best practices (Clauset et al., 2009; Corral et al., 2015):
- **MLE fitting** instead of OLS on log-log regressions (which introduce significant bias)
- **Kolmogorov-Smirnov (KS) statistic** for goodness-of-fit
- **AIC/BIC** for model comparison
- **Residual analysis by frequency bins** (head/mid/tail) to identify systematic deviations

## Results Summary

| Model | Parameters | Log-Likelihood | KS Statistic | AIC | BIC | R² (log-log) |
|-------|------------|----------------|--------------|-----|-----|--------------|
| Zipf | s=1.0010, b=0 | -783,709.46 | 0.1012 | 1,567,420.92 | 1,567,427.66 | 0.7560 |
| Zipf-Mandelbrot | s=1.2710, b=8.2490 | -772,899.63 | **0.0114** | **1,545,803.26** | **1,545,816.74** | **0.9659** |

**Model Comparison**: ΔAIC = -21,617.66, ΔBIC = -21,610.92 → **Zipf-Mandelbrot strongly preferred**

## Key Findings

1. **Zipf-Mandelbrot dramatically outperforms Zipf**: The 2-parameter model provides substantially better fit (KS=0.011 vs 0.101, R²=0.966 vs 0.756)
2. **Curvature parameter b=8.249**: The positive curvature indicates the frequency distribution bends downward at high ranks, which the standard Zipf law (b=0) cannot capture
3. **Residual analysis reveals systematic deviations**: 
   - Zipf shows large positive residuals in the head (top 10% ranks: mean=14.2, SD=379) and systematic negative residuals in mid/tail
   - Zipf-Mandelbrot residuals are near-zero across all bins (head: 0.09, mid: 0.007, tail: -0.15)

## Files

- `eval.py` — Main evaluation script (MLE fitting, KS, AIC/BIC, residual analysis)
- `eval_out.json` — Schema-compliant results (exp_eval_sol_out format)
- `mini_eval_out.json` — Mini version (3 examples)
- `preview_eval_out.json` — Preview version (truncated strings)
- `figures/fig_data.json` — Raw data for figure generation (6271 ranks × frequencies)
- `figures/fig1_loglog.pdf/png` — Rank-frequency distribution on log-log axes
- `figures/fig2_residuals.pdf/png` — Residuals vs fitted values for both models
- `figures/fig3_model_comparison.pdf/png` — Bar chart of KS, AIC, BIC, R²
- `figures/fig4_residual_bins.pdf/png` — Residual analysis by rank bin (head/mid/tail)
- `figures/fig5_panel.pdf/png` — Multi-panel composite figure
- `logs/run.log` — Execution log

## Running the Evaluation

```bash
cd /Volumes/aii-s20020/work/run_fODCbnzS2ZPQ/round-2/evaluation-2/src
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install numpy scipy loguru matplotlib
python eval.py
```

## Restoring Removed Files

The `.venv/` directory is excluded from the manifest (regenerable):

```bash
uv venv .venv --python=3.12
uv pip install numpy scipy loguru matplotlib
```

## Dependencies

- Python 3.12+
- numpy, scipy, loguru, matplotlib
- Input text: `/Volumes/aii-s20020/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/pride_and_prejudice_clean.txt`

## References

- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. SIAM Review, 51(4), 661-703.
- Corral, Á., et al. (2015). Universal complexity of Zipf's law. Physical Review E, 91(3), 032117.