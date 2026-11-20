
import numpy as np
import scipy.optimize as opt
from collections import Counter
import re
import json

def get_word_frequencies(text):
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    frequencies = sorted(counts.values(), reverse=True)
    return np.array(frequencies)

def log_likelihood(params, ranks, freqs):
    s, b = params
    if s <= 0 or b < 0:
        return 1e10  # Penalize invalid parameters
    
    # Probabilities
    probs = (ranks + b) ** (-s)
    probs /= np.sum(probs)
    
    # Log-likelihood: sum(freqs * log(probs))
    ll = np.sum(freqs * np.log(probs + 1e-10))
    return -ll # Minimize negative log-likelihood

def fit_zipf_mandelbrot(freqs):
    ranks = np.arange(1, len(freqs) + 1)
    # Initial guess: s=1, b=1
    result = opt.minimize(log_likelihood, [1.0, 1.0], args=(ranks, freqs), method='Nelder-Mead')
    return result.x

def fit_zipf(freqs):
    ranks = np.arange(1, len(freqs) + 1)
    # b is fixed to 0. Optimize only s.
    def ll_zipf(s, ranks, freqs):
        if s <= 0: return 1e10
        probs = ranks ** (-s)
        probs /= np.sum(probs)
        return -np.sum(freqs * np.log(probs + 1e-10))
    
    result = opt.minimize(ll_zipf, [1.0], args=(ranks, freqs), method='Nelder-Mead')
    return result.x[0], 0.0

def calculate_ks(freqs, s, b):
    ranks = np.arange(1, len(freqs) + 1)
    # Theoretical frequencies
    probs = (ranks + b) ** (-s)
    probs /= np.sum(probs)
    
    # Cumulative distribution
    obs_cdf = np.cumsum(freqs) / np.sum(freqs)
    theo_cdf = np.cumsum(probs)
    
    return np.max(np.abs(obs_cdf - theo_cdf))

def calculate_aic_bic(ll, k, n):
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return aic, bic

def calculate_r2(freqs, s, b):
    """R² on log-log rank-frequency space."""
    ranks = np.arange(1, len(freqs) + 1)
    probs = (ranks + b) ** (-s)
    probs /= np.sum(probs)
    theo_freqs = probs * np.sum(freqs)
    
    log_obs = np.log10(freqs)
    log_theo = np.log10(theo_freqs)
    
    ss_res = np.sum((log_obs - log_theo) ** 2)
    ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    return 1 - ss_res / ss_tot

def residual_by_bin(freqs, s, b, n_bins=10):
    """Residual analysis binned by frequency (head/mid/tail)."""
    ranks = np.arange(1, len(freqs) + 1)
    probs = (ranks + b) ** (-s)
    probs /= np.sum(probs)
    theo_freqs = probs * np.sum(freqs)
    
    # Bin by rank (which correlates with frequency)
    bins = np.array_split(np.arange(len(freqs)), n_bins)
    residuals = []
    for i, bin_idx in enumerate(bins):
        obs_mean = np.mean(freqs[bin_idx])
        theo_mean = np.mean(theo_freqs[bin_idx])
        residuals.append({
            "bin": i,
            "rank_start": int(bin_idx[0] + 1),
            "rank_end": int(bin_idx[-1] + 1),
            "observed_mean": float(obs_mean),
            "fitted_mean": float(theo_mean),
            "residual": float(obs_mean - theo_mean)
        })
    return residuals

def main():
    with open('pride_and_prejudice_clean.txt', 'r') as f:
        text = f.read()
    
    freqs = get_word_frequencies(text)
    n = np.sum(freqs)
    
    # Fit Zipf-Mandelbrot
    s_zm, b_zm = fit_zipf_mandelbrot(freqs)
    # Fit Zipf
    s_z, b_z = fit_zipf(freqs)
    
    # Calculate Metrics
    ranks = np.arange(1, len(freqs) + 1)
    
    # Log-likelihoods
    ll_zm = -log_likelihood([s_zm, b_zm], ranks, freqs)
    ll_z = -log_likelihood([s_z, b_z], ranks, freqs)
    
    aic_zm, bic_zm = calculate_aic_bic(ll_zm, 2, n)
    aic_z, bic_z = calculate_aic_bic(ll_z, 1, n)
    
    ks_zm = calculate_ks(freqs, s_zm, b_zm)
    ks_z = calculate_ks(freqs, s_z, b_z)
    
    r2_zm = calculate_r2(freqs, s_zm, b_zm)
    r2_z = calculate_r2(freqs, s_z, b_z)
    
    res_zm = residual_by_bin(freqs, s_zm, b_zm)
    res_z = residual_by_bin(freqs, s_z, b_z)
    
    # Create the schema-compliant output
    results = {
        "metadata": {
            "evaluation_name": "Global Zipfian Fit Accuracy Analysis",
            "description": "Rigorous global MLE-based fitting of Zipf and Zipf-Mandelbrot models to Pride and Prejudice",
            "parameters": {
                "source_text": "pride_and_prejudice_clean.txt",
                "method": "Maximum Likelihood Estimation"
            }
        },
        "metrics_agg": {
            "zipf_mandelbrot_s": float(s_zm),
            "zipf_mandelbrot_b": float(b_zm),
            "zipf_mandelbrot_aic": float(aic_zm),
            "zipf_mandelbrot_bic": float(bic_zm),
            "zipf_mandelbrot_ks": float(ks_zm),
            "zipf_mandelbrot_r2": float(r2_zm),
            "zipf_s": float(s_z),
            "zipf_b": float(b_z),
            "zipf_aic": float(aic_z),
            "zipf_bic": float(bic_z),
            "zipf_ks": float(ks_z),
            "zipf_r2": float(r2_z),
            "aic_difference": float(aic_z - aic_zm),  # positive means ZM better
            "bic_difference": float(bic_z - bic_zm),
            "ks_improvement": float(ks_z - ks_zm)  # positive means ZM better
        },
        "datasets": [
            {
                "dataset": "pride_and_prejudice",
                "examples": [
                    {
                        "input": "Evaluate Zipf-Mandelbrot model fit on Pride and Prejudice word frequencies",
                        "output": "Model fit completed with MLE estimation",
                        "predict_s": str(s_zm),
                        "predict_b": str(b_zm),
                        "predict_aic": str(aic_zm),
                        "predict_bic": str(bic_zm),
                        "predict_ks": str(ks_zm),
                        "predict_r2": str(r2_zm),
                        "eval_ks": float(ks_zm),
                        "eval_r2": float(r2_zm),
                        "eval_aic": float(aic_zm),
                        "eval_bic": float(bic_zm),
                        "metadata_residuals_by_bin": json.dumps(res_zm)
                    },
                    {
                        "input": "Evaluate Zipf model fit on Pride and Prejudice word frequencies",
                        "output": "Model fit completed with MLE estimation",
                        "predict_s": str(s_z),
                        "predict_b": str(b_z),
                        "predict_aic": str(aic_z),
                        "predict_bic": str(bic_z),
                        "predict_ks": str(ks_z),
                        "predict_r2": str(r2_z),
                        "eval_ks": float(ks_z),
                        "eval_r2": float(r2_z),
                        "eval_aic": float(aic_z),
                        "eval_bic": float(bic_z),
                        "metadata_residuals_by_bin": json.dumps(res_z)
                    }
                ]
            }
        ]
    }
    
    with open('exp_eval_sol_out.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
