# Global Zipf Fit on Pride and Prejudice

This experiment implements Maximum Likelihood Estimation (MLE) fitting of Zipf and Zipf-Mandelbrot models to global word frequencies from Pride and Prejudice (Project Gutenberg eBook #42671), evaluating goodness-of-fit with R², KS statistic, AIC/BIC, and bootstrap confidence intervals.

## Results Summary

- **Novel statistics**: 123,350 tokens, 6,425 unique word types
- **Zipf model**: s = 0.992 [95% CI: 0.991, 0.995], KS = 0.093, R² = 0.739
- **Zipf-Mandelbrot model**: s = 1.267 [95% CI: 1.258, 1.269], b = 8.165 [95% CI: 7.943, 8.410], KS = 0.011, R² = 0.966
- **Model comparison**: Zipf-Mandelbrot strongly preferred (ΔAIC = 21,462)

The Zipf-Mandelbrot model provides substantially better fit, especially in the head and body of the distribution.

## Files

- `method.py` - Main experimental script implementing MLE fitting, goodness-of-fit metrics, residual analysis, and bootstrap CIs
- `generate_figures.py` - Script to generate publication-quality figures
- `exp_gen_sol_out.json` - Primary output in standardized schema format
- `results.json` - Raw results for easy access
- `figures/` - Rendered figures (PDF and PNG)
  - `fig1_rank_freq.pdf/png` - Log-log rank-frequency plot with both fits
  - `fig2_residuals.pdf/png` - Mean residuals by frequency rank region
  - `fig3_gof.pdf/png` - Goodness-of-fit comparison
  - `fig4_bootstrap_ci.pdf/png` - Bootstrap confidence intervals for parameters
- `full_*/mini_*/preview_*` - JSON variants for different use cases

## How to Run

```bash
# Run the main experiment
uv run method.py

# Generate figures
uv run generate_figures.py
```

## Restoring Removed Files

The following files can be regenerated if deleted:

- `.venv/` - Python virtual environment with all dependencies
  ```bash
  uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r pyproject.toml
  ```