# Global Zipfian Fit Accuracy Analysis

This artifact evaluates how accurately Zipf's law and the Zipf-Mandelbrot model describe word frequencies in Pride and Prejudice (Project Gutenberg eBook #42671).

## Methodology

Following computational linguistics best practices (Clauset et al., 2009; Corral et al., 2015):
- **MLE fitting** instead of OLS on log-log regressions (which introduce significant bias)
- **Kolmogorov-Smirnov (KS) statistic** for goodness-of-fit
- **AIC/BIC** for model comparison
- **Residual analysis by frequency bins** (head/mid/tail) to identify systematic deviations

## Results Summary

| Model | Exponent (s) | Curvature (b) | AIC | BIC | KS | R² |
|-------|-------------|---------------|-----|-----|-----|-----|
| Zipf (b=0) | 0.992 | 0.0 | 1,581,276 | 1,581,286 | 0.093 | 0.741 |
| Zipf-Mandelbrot | 1.266 | 8.138 | 1,559,827 | 1,559,847 | 0.011 | 0.967 |

**Key finding**: Zipf-Mandelbrot dramatically outperforms standard Zipf:
- ΔAIC = 21,449 (strongly favors ZM)
- ΔBIC = 21,440 (strongly favors ZM)
- KS improves 8.4× (0.093 → 0.011)
- R² improves from 0.74 to 0.97

Residual analysis shows Zipf-Mandelbrot captures the head of the distribution well (residuals ≈ 0), while standard Zipf systematically underestimates high-frequency words (residuals ~15).

## Files

| File | Description |
|------|-------------|
| `eval.py` | Main evaluation script |
| `exp_eval_sol_out.json` | Full evaluation results (schema-compliant) |
| `full_exp_eval_sol_out.json` | Full version for pipeline |
| `mini_exp_eval_sol_out.json` | Mini version (3 examples) |
| `preview_exp_eval_sol_out.json` | Preview version (truncated strings) |
| `fig_model_comparison.json/.pdf/.png` | Model comparison bar chart |
| `fig_residuals.json/.pdf/.png` | Residual-by-bin horizontal bar chart |
| `fig_rank_freq.json/.pdf/.png` | Rank-frequency scatter with model fits |
| `pride_and_prejudice_clean.txt` | Source text (676 KB) |

## Running the Evaluation

```bash
uv venv .venv --python=3.12
uv pip install -e .
python eval.py
```

## Restoring Removed Files

```bash
# Virtual environment
uv venv .venv --python=3.12 && uv pip install -e .

# Note: pride_and_prejudice_clean.txt is kept (source text)
# Figures are kept as they are the evaluation deliverables
```