#!/usr/bin/env python3
"""
Independent verification script for headline numbers.
Reads raw text, recomputes frequency distribution, fits models, and verifies results.
"""

import json
import numpy as np
import scipy.optimize as opt
from collections import Counter
import re
from pathlib import Path

def clean_text(text: str) -> str:
    start_idx = text.find("PRIDE AND PREJUDICE:")
    if start_idx != -1:
        text = text[start_idx:]
    end_idx = text.find("Transcriber's note:")
    if end_idx != -1:
        text = text[:end_idx]
    return text

def tokenize(text: str) -> list:
    text = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words

def get_frequency_distribution(words: list) -> tuple:
    counts = Counter(words)
    freqs = np.array([c for _, c in counts.most_common()])
    ranks = np.arange(1, len(freqs) + 1)
    return ranks, freqs

def zipf_log_likelihood(s, ranks, freqs):
    if s <= 1.0:
        return np.inf
    probs = 1.0 / (ranks ** s)
    probs /= np.sum(probs)
    ll = np.sum(freqs * np.log(probs))
    return -ll

def zipf_mandelbrot_log_likelihood(params, ranks, freqs):
    s, b = params
    if s <= 0 or b < 0:
        return np.inf
    probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    ll = np.sum(freqs * np.log(probs))
    return -ll

def fit_zipf(ranks, freqs):
    result = opt.minimize_scalar(
        lambda s: zipf_log_likelihood(s, ranks, freqs),
        bounds=(1.001, 5.0),
        method='bounded'
    )
    return result.x, result.fun

def fit_zipf_mandelbrot(ranks, freqs):
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
    cumsum_obs = np.cumsum(freqs)
    cdf_obs = cumsum_obs / cumsum_obs[-1]
    
    if b == 0:
        probs = 1.0 / (ranks ** s)
    else:
        probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    cdf_model = np.cumsum(probs)
    
    ks = np.max(np.abs(cdf_obs - cdf_model))
    return ks

def calculate_aic_bic(n_params, log_likelihood, n_obs):
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood
    return aic, bic

def calculate_r_squared(ranks, freqs, s, b=0):
    log_obs = np.log(freqs)
    
    if b == 0:
        probs = 1.0 / (ranks ** s)
    else:
        probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    pred_freqs = probs * np.sum(freqs)
    log_pred = np.log(pred_freqs)
    
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r2

def residual_bins(ranks, freqs, s, b=0):
    if b == 0:
        probs = 1.0 / (ranks ** s)
    else:
        probs = 1.0 / ((ranks + b) ** s)
    probs /= np.sum(probs)
    pred_freqs = probs * np.sum(freqs)
    
    residuals = freqs - pred_freqs
    
    n = len(ranks)
    head_end = max(1, n // 10)
    tail_start = max(head_end + 1, n * 9 // 10)
    
    head_mask = ranks <= head_end
    tail_mask = ranks >= tail_start
    mid_mask = (~head_mask) & (~tail_mask)
    
    bins = {}
    if np.any(head_mask):
        bins['head'] = {
            'mean': float(np.mean(residuals[head_mask])),
            'std': float(np.std(residuals[head_mask])),
            'n': int(np.sum(head_mask))
        }
    if np.any(mid_mask):
        bins['mid'] = {
            'mean': float(np.mean(residuals[mid_mask])),
            'std': float(np.std(residuals[mid_mask])),
            'n': int(np.sum(mid_mask))
        }
    if np.any(tail_mask):
        bins['tail'] = {
            'mean': float(np.mean(residuals[tail_mask])),
            'std': float(np.std(residuals[tail_mask])),
            'n': int(np.sum(tail_mask))
        }
    
    return bins

def main():
    # Load text
    text_path = Path("/Volumes/aii-s20003/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/pride_and_prejudice_clean.txt")
    raw_text = text_path.read_text()
    text = clean_text(raw_text)
    words = tokenize(text)
    
    print(f"Tokens: {len(words):,}, Types: {len(set(words)):,}")
    
    ranks, freqs = get_frequency_distribution(words)
    print(f"Ranks: {len(ranks)}")
    
    # Fit Zipf
    s_zipf, neg_ll_zipf = fit_zipf(ranks, freqs)
    ll_zipf = -neg_ll_zipf
    
    # Fit ZM
    s_zm, b_zm, neg_ll_zm = fit_zipf_mandelbrot(ranks, freqs)
    ll_zm = -neg_ll_zm
    
    # KS
    ks_zipf = calculate_ks_statistic(ranks, freqs, s_zipf, b=0)
    ks_zm = calculate_ks_statistic(ranks, freqs, s_zm, b=b_zm)
    
    # AIC/BIC
    n_obs = len(freqs)
    aic_zipf, bic_zipf = calculate_aic_bic(1, ll_zipf, n_obs)
    aic_zm, bic_zm = calculate_aic_bic(2, ll_zm, n_obs)
    
    # R²
    r2_zipf = calculate_r_squared(ranks, freqs, s_zipf, b=0)
    r2_zm = calculate_r_squared(ranks, freqs, s_zm, b=b_zm)
    
    # Residual bins
    res_zipf = residual_bins(ranks, freqs, s_zipf, b=0)
    res_zm = residual_bins(ranks, freqs, s_zm, b=b_zm)
    
    # Print results
    print("\n=== VERIFICATION RESULTS ===")
    print(f"Zipf: s={s_zipf:.4f}, LL={ll_zipf:.2f}, KS={ks_zipf:.4f}, AIC={aic_zipf:.2f}, BIC={bic_zipf:.2f}, R²={r2_zipf:.4f}")
    print(f"ZM:   s={s_zm:.4f}, b={b_zm:.4f}, LL={ll_zm:.2f}, KS={ks_zm:.4f}, AIC={aic_zm:.2f}, BIC={bic_zm:.2f}, R²={r2_zm:.4f}")
    print(f"ΔAIC = {aic_zm - aic_zipf:.2f}, ΔBIC = {bic_zm - bic_zipf:.2f}")
    print(f"Residuals Zipf: head={res_zipf['head']['mean']:.2f}, mid={res_zipf['mid']['mean']:.2f}, tail={res_zipf['tail']['mean']:.2f}")
    print(f"Residuals ZM:   head={res_zm['head']['mean']:.2f}, mid={res_zm['mid']['mean']:.2f}, tail={res_zm['tail']['mean']:.2f}")
    
    # Load reference results (both old and new format)
    with open("exp_eval_sol_out.json") as f:
        ref = json.load(f)
    
    # New format: metrics_agg
    m = ref.get("metrics_agg", {})
    
    print("\n=== COMPARISON WITH REFERENCE (new format) ===")
    print(f"s_zipf:  computed={s_zipf:.4f}, ref={m.get('zipf_s', 'N/A')}, match={abs(s_zipf - m.get('zipf_s', s_zipf)) < 0.01}")
    print(f"s_zm:    computed={s_zm:.4f}, ref={m.get('zipf_mandelbrot_s', 'N/A')}, match={abs(s_zm - m.get('zipf_mandelbrot_s', s_zm)) < 0.01}")
    print(f"b_zm:    computed={b_zm:.4f}, ref={m.get('zipf_mandelbrot_b', 'N/A')}, match={abs(b_zm - m.get('zipf_mandelbrot_b', b_zm)) < 0.01}")
    print(f"KS_zipf: computed={ks_zipf:.4f}, ref={m.get('zipf_ks', 'N/A')}, match={abs(ks_zipf - m.get('zipf_ks', ks_zipf)) < 0.01}")
    print(f"KS_zm:   computed={ks_zm:.4f}, ref={m.get('zipf_mandelbrot_ks', 'N/A')}, match={abs(ks_zm - m.get('zipf_mandelbrot_ks', ks_zm)) < 0.01}")
    print(f"R2_zipf: computed={r2_zipf:.4f}, ref={m.get('zipf_r2', 'N/A')}, match={abs(r2_zipf - m.get('zipf_r2', r2_zipf)) < 0.01}")
    print(f"R2_zm:   computed={r2_zm:.4f}, ref={m.get('zipf_mandelbrot_r2', 'N/A')}, match={abs(r2_zm - m.get('zipf_mandelbrot_r2', r2_zm)) < 0.01}")
    print(f"ΔAIC:    computed={aic_zm - aic_zipf:.2f}, ref={m.get('aic_difference', 'N/A')}")
    
    # Check qualitative conclusion
    print("\n=== QUALITATIVE CONCLUSION ===")
    print(f"ZM preferred (lower AIC): {aic_zm < aic_zipf}")
    print(f"ZM preferred (lower BIC): {bic_zm < bic_zipf}")
    print(f"ZM preferred (lower KS):  {ks_zm < ks_zipf}")
    print(f"ZM preferred (higher R²): {r2_zm > r2_zipf}")
    print("ALL CRITERIA FAVOR ZIPF-MANDELBROT:", all([aic_zm < aic_zipf, bic_zm < bic_zipf, ks_zm < ks_zipf, r2_zm > r2_zipf]))
    
    # Also check old format for comparison
    if "datasets" in ref:
        with open("eval_out.json") as f:
            old_ref = json.load(f)
        if "datasets" in old_ref and old_ref["datasets"]:
            try:
                old_output = json.loads(old_ref["datasets"][0]["examples"][0]["output"])
                print("\n=== COMPARISON WITH OLD REFERENCE ===")
                print(f"Old s_zipf:  {old_output['zipf_model']['s']:.4f}")
                print(f"Old s_zm:    {old_output['zipf_mandelbrot_model']['s']:.4f}")
            except:
                pass

if __name__ == "__main__":
    main()