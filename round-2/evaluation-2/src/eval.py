#!/usr/bin/env python3
"""
Global Zipfian Fit Accuracy Analysis

Rigorous MLE-based fitting of Zipf and Zipf-Mandelbrot models to Pride and Prejudice,
including goodness-of-fit (KS statistic), model comparison (AIC/BIC), and residual analysis.
"""

import json
import numpy as np
import scipy.optimize as opt
from scipy import stats
from collections import Counter
from loguru import logger
from pathlib import Path
import sys
import re

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


def clean_text(text: str) -> str:
    """Remove Project Gutenberg boilerplate and clean text."""
    # Remove everything before "PRIDE AND PREJUDICE:"
    start_idx = text.find("PRIDE AND PREJUDICE:")
    if start_idx != -1:
        text = text[start_idx:]
    # Remove transcriber notes at the end
    end_idx = text.find("Transcriber's note:")
    if end_idx != -1:
        text = text[:end_idx]
    return text


def tokenize(text: str) -> list:
    """Tokenize text into words (lowercase, alphanumeric only)."""
    text = text.lower()
    # Use regex to extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words


def get_frequency_distribution(words: list) -> tuple:
    """Get rank-frequency distribution from word list."""
    counts = Counter(words)
    freqs = np.array([c for _, c in counts.most_common()])
    ranks = np.arange(1, len(freqs) + 1)
    return ranks, freqs


def zipf_log_likelihood(s, ranks, freqs):
    """Negative log-likelihood for Zipf distribution (b=0)."""
    if s <= 1.0:
        return np.inf
    probs = 1.0 / (ranks ** s)
    probs /= np.sum(probs)
    ll = np.sum(freqs * np.log(probs))
    return -ll


def zipf_mandelbrot_log_likelihood(params, ranks, freqs):
    """Negative log-likelihood for Zipf-Mandelbrot distribution."""
    s, b = params
    if s <= 0 or b < 0:
        return np.inf
    probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    ll = np.sum(freqs * np.log(probs))
    return -ll


def fit_zipf(ranks, freqs):
    """Fit Zipf model via MLE (s only, b=0)."""
    initial_guess = 1.5
    result = opt.minimize_scalar(
        lambda s: zipf_log_likelihood(s, ranks, freqs),
        bounds=(1.001, 5.0),
        method='bounded'
    )
    return result.x, result.fun


def fit_zipf_mandelbrot(ranks, freqs):
    """Fit Zipf-Mandelbrot model via MLE (s and b)."""
    initial_guess = [1.2, 2.0]
    result = opt.minimize(
        zipf_mandelbrot_log_likelihood,
        initial_guess,
        args=(ranks, freqs),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    return result.x[0], result.x[1], result.fun


def calculate_ks_statistic(ranks, freqs, s, b=0):
    """Calculate Kolmogorov-Smirnov statistic for goodness of fit."""
    # Empirical CDF
    cumsum_obs = np.cumsum(freqs)
    cdf_obs = cumsum_obs / cumsum_obs[-1]
    
    # Model CDF
    if b == 0:
        probs = 1.0 / (ranks ** s)
    else:
        probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    cdf_model = np.cumsum(probs)
    
    # KS statistic
    ks = np.max(np.abs(cdf_obs - cdf_model))
    return ks


def calculate_aic_bic(n_params, log_likelihood, n_obs):
    """Calculate AIC and BIC."""
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood
    return aic, bic


def calculate_r_squared(ranks, freqs, s, b=0):
    """Calculate R² on log-log rank-frequency space."""
    # Observed log frequencies
    log_obs = np.log(freqs)
    
    # Predicted log frequencies
    if b == 0:
        probs = 1.0 / (ranks ** s)
    else:
        probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    pred_freqs = probs * np.sum(freqs)
    log_pred = np.log(pred_freqs)
    
    # R²
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r2


def residual_analysis(ranks, freqs, s, b=0):
    """Calculate residuals by frequency bins (head/tail/mid-rank)."""
    # Calculate predicted frequencies
    if b == 0:
        probs = 1.0 / (ranks ** s)
    else:
        probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    pred_freqs = probs * np.sum(freqs)
    
    # Residuals (observed - predicted)
    residuals = freqs - pred_freqs
    
    # Bin by rank (head, mid, tail)
    n = len(ranks)
    head_end = max(1, n // 10)
    tail_start = max(head_end + 1, n * 9 // 10)
    
    head_mask = ranks <= head_end
    tail_mask = ranks >= tail_start
    mid_mask = (~head_mask) & (~tail_mask)
    
    bins = {}
    if np.any(head_mask):
        bins['head'] = {
            'mean_residual': float(np.mean(residuals[head_mask])),
            'std_residual': float(np.std(residuals[head_mask])),
            'n': int(np.sum(head_mask))
        }
    if np.any(mid_mask):
        bins['mid'] = {
            'mean_residual': float(np.mean(residuals[mid_mask])),
            'std_residual': float(np.std(residuals[mid_mask])),
            'n': int(np.sum(mid_mask))
        }
    if np.any(tail_mask):
        bins['tail'] = {
            'mean_residual': float(np.mean(residuals[tail_mask])),
            'std_residual': float(np.std(residuals[tail_mask])),
            'n': int(np.sum(tail_mask))
        }
    
    return bins


@logger.catch(reraise=True)
def main():
    # Load the clean text
    text_path = Path("/Volumes/aii-s20020/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/pride_and_prejudice_clean.txt")
    if not text_path.exists():
        logger.error(f"Text file not found at {text_path}")
        return
    
    logger.info(f"Loading text from {text_path}")
    raw_text = text_path.read_text()
    
    logger.info("Cleaning text...")
    text = clean_text(raw_text)
    
    logger.info("Tokenizing...")
    words = tokenize(text)
    logger.info(f"Total tokens: {len(words)}")
    logger.info(f"Unique tokens: {len(set(words))}")
    
    logger.info("Computing frequency distribution...")
    ranks, freqs = get_frequency_distribution(words)
    logger.info(f"Number of ranks: {len(ranks)}")
    
    # Fit Zipf model (b=0)
    logger.info("Fitting Zipf model (MLE)...")
    s_zipf, neg_ll_zipf = fit_zipf(ranks, freqs)
    ll_zipf = -neg_ll_zipf
    
    # Fit Zipf-Mandelbrot model
    logger.info("Fitting Zipf-Mandelbrot model (MLE)...")
    s_zm, b_zm, neg_ll_zm = fit_zipf_mandelbrot(ranks, freqs)
    ll_zm = -neg_ll_zm
    
    # Goodness of fit - KS statistic
    logger.info("Calculating KS statistics...")
    ks_zipf = calculate_ks_statistic(ranks, freqs, s_zipf, b=0)
    ks_zm = calculate_ks_statistic(ranks, freqs, s_zm, b=b_zm)
    
    # Model comparison - AIC/BIC
    logger.info("Calculating AIC/BIC...")
    n_obs = len(freqs)
    aic_zipf, bic_zipf = calculate_aic_bic(1, ll_zipf, n_obs)
    aic_zm, bic_zm = calculate_aic_bic(2, ll_zm, n_obs)
    
    # R² on log-log
    logger.info("Calculating R²...")
    r2_zipf = calculate_r_squared(ranks, freqs, s_zipf, b=0)
    r2_zm = calculate_r_squared(ranks, freqs, s_zm, b=b_zm)
    
    # Residual analysis
    logger.info("Performing residual analysis...")
    res_zipf = residual_analysis(ranks, freqs, s_zipf, b=0)
    res_zm = residual_analysis(ranks, freqs, s_zm, b=b_zm)
    
    # Compile results for schema-compliant output
    # The exp_eval_sol_out schema requires: metrics_agg, datasets with examples containing input/output strings
    
    # Aggregate metrics
    metrics_agg = {
        "n_tokens": float(len(words)),
        "n_types": float(len(set(words))),
        "n_ranks": float(len(ranks)),
        "zipf_s": float(s_zipf),
        "zipf_ks": float(ks_zipf),
        "zipf_aic": float(aic_zipf),
        "zipf_bic": float(bic_zipf),
        "zipf_r2": float(r2_zipf),
        "zipf_log_likelihood": float(ll_zipf),
        "zm_s": float(s_zm),
        "zm_b": float(b_zm),
        "zm_ks": float(ks_zm),
        "zm_aic": float(aic_zm),
        "zm_bic": float(bic_zm),
        "zm_r2": float(r2_zm),
        "zm_log_likelihood": float(ll_zm),
        "delta_aic": float(aic_zm - aic_zipf),
        "delta_bic": float(bic_zm - bic_zipf)
    }
    
    # Example input and output as JSON strings
    input_str = json.dumps({
        "dataset": "pride_and_prejudice",
        "task": "global_zipfian_fit",
        "models": ["zipf", "zipf_mandelbrot"],
        "fitting_method": "MLE"
    })
    
    output_str = json.dumps({
        "zipf_model": {
            "s": float(s_zipf),
            "b": 0.0,
            "log_likelihood": float(ll_zipf),
            "ks_statistic": float(ks_zipf),
            "aic": float(aic_zipf),
            "bic": float(bic_zipf),
            "r2_loglog": float(r2_zipf),
            "residual_bins": res_zipf
        },
        "zipf_mandelbrot_model": {
            "s": float(s_zm),
            "b": float(b_zm),
            "log_likelihood": float(ll_zm),
            "ks_statistic": float(ks_zm),
            "aic": float(aic_zm),
            "bic": float(bic_zm),
            "r2_loglog": float(r2_zm),
            "residual_bins": res_zm
        },
        "model_comparison": {
            "delta_aic": float(aic_zm - aic_zipf),
            "delta_bic": float(bic_zm - bic_zipf),
            "preferred_model": "zipf_mandelbrot" if aic_zm < aic_zipf else "zipf"
        }
    })
    
    # Per-example evaluation metrics
    eval_metrics = {
        "eval_zipf_ks": float(ks_zipf),
        "eval_zm_ks": float(ks_zm),
        "eval_delta_aic": float(aic_zm - aic_zipf),
        "eval_delta_bic": float(bic_zm - bic_zipf),
        "eval_preferred_zm": 1.0 if aic_zm < aic_zipf else 0.0
    }
    
    # Create schema-compliant output
    output = {
        "metadata": {
            "evaluation_name": "global_zipfian_fit_accuracy",
            "description": "Global MLE-based fitting of Zipf and Zipf-Mandelbrot models to Pride and Prejudice with KS, AIC/BIC, and residual analysis",
            "parameters": {
                "tokenization": "regex [a-zA-Z]+ lowercase",
                "fitting_method": "MLE via Nelder-Mead (ZM) and bounded scalar (Zipf)",
                "goodness_of_fit": "KS statistic",
                "model_comparison": "AIC/BIC",
                "residual_bins": "head (top 10%), mid, tail (bottom 10%)"
            }
        },
        "metrics_agg": metrics_agg,
        "datasets": [{
            "dataset": "pride_and_prejudice",
            "examples": [{
                "input": input_str,
                "output": output_str,
                "metadata_dataset": "pride_and_prejudice",
                "metadata_n_tokens": len(words),
                "metadata_n_types": len(set(words)),
                "metadata_n_ranks": len(ranks),
                **eval_metrics
            }]
        }]
    }
    
    # Save output
    Path("eval_out.json").write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to eval_out.json")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tokens: {len(words):,}, Types: {len(set(words)):,}")
    logger.info(f"Zipf: s={s_zipf:.4f}, LL={ll_zipf:.2f}, KS={ks_zipf:.4f}, AIC={aic_zipf:.2f}, BIC={bic_zipf:.2f}, R²={r2_zipf:.4f}")
    logger.info(f"Zipf-Mandelbrot: s={s_zm:.4f}, b={b_zm:.4f}, LL={ll_zm:.2f}, KS={ks_zm:.4f}, AIC={aic_zm:.2f}, BIC={bic_zm:.2f}, R²={r2_zm:.4f}")
    logger.info(f"ΔAIC = {aic_zm - aic_zipf:.2f}, ΔBIC = {bic_zm - bic_zipf:.2f}")
    logger.info(f"Preferred model (AIC): {'zipf_mandelbrot' if aic_zm < aic_zipf else 'zipf'}")
    
    # Save raw data for figure generation
    # Compute predicted frequencies for figure
    ranks_list = ranks.tolist()
    freqs_list = freqs.tolist()
    
    # Zipf predicted
    probs_zipf = 1.0 / (ranks ** s_zipf)
    probs_zipf /= np.sum(probs_zipf)
    pred_zipf = (probs_zipf * np.sum(freqs)).tolist()
    
    # ZM predicted
    probs_zm = 1.0 / ((ranks + b_zm) ** s_zm)
    probs_zm /= np.sum(probs_zm)
    pred_zm = (probs_zm * np.sum(freqs)).tolist()
    
    # Residuals
    resid_zipf = (freqs - pred_zipf).tolist()
    resid_zm = (freqs - pred_zm).tolist()
    
    fig_data = {
        "ranks": ranks_list,
        "observed_freqs": freqs_list,
        "zipf_predicted": pred_zipf,
        "zm_predicted": pred_zm,
        "zipf_residuals": resid_zipf,
        "zm_residuals": resid_zm,
        "s_zipf": s_zipf,
        "b_zipf": 0.0,
        "s_zm": s_zm,
        "b_zm": b_zm,
        "ks_zipf": ks_zipf,
        "ks_zm": ks_zm,
        "r2_zipf": r2_zipf,
        "r2_zm": r2_zm
    }
    Path("figures/fig_data.json").write_text(json.dumps(fig_data, indent=2))
    logger.info("Saved figure data to figures/fig_data.json")


if __name__ == "__main__":
    main()
