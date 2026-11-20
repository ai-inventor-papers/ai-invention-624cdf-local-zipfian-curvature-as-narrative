import json
import re
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from scipy.stats import ks_1samp
import argparse
from pathlib import Path

# Zipf-Mandelbrot: p(r) = C / (r + b)^s
# Zipf: p(r) = C / r^s
# where C = 1 / sum(1 / (r + b)^s) for r=1...N

def zipf_mandelbrot_pdf(s, b, n):
    r_arr = np.arange(1, n + 1)
    probs = 1.0 / np.power(r_arr + b, s)
    return probs / np.sum(probs)

def zipf_pdf(s, n):
    r_arr = np.arange(1, n + 1)
    probs = 1.0 / np.power(r_arr, s)
    return probs / np.sum(probs)

def log_likelihood_zm(params, freqs):
    s, b = params
    n = len(freqs)
    probs = zipf_mandelbrot_pdf(s, b, n)
    probs = np.maximum(probs, 1e-10)
    return -np.sum(freqs * np.log(probs))

def log_likelihood_zipf(s, freqs):
    n = len(freqs)
    probs = zipf_pdf(s, n)
    probs = np.maximum(probs, 1e-10)
    return -np.sum(freqs * np.log(probs))

def fit_zipf(freqs):
    n = len(freqs)
    res = minimize(
        lambda s: log_likelihood_zipf(s[0], freqs),
        [1.0],
        bounds=[(0.01, 5.0)],
        method='L-BFGS-B'
    )
    s = res.x[0]
    nll = res.fun
    aic = 2 * 1 + 2 * nll
    return s, nll, aic

def fit_zipf_mandelbrot(freqs):
    n = len(freqs)
    res = minimize(
        lambda p: log_likelihood_zm(p, freqs),
        [1.0, 1.0],
        bounds=[(0.01, 5.0), (0.0, 100.0)],
        method='L-BFGS-B'
    )
    s, b = res.x[0], res.x[1]
    nll = res.fun
    aic = 2 * 2 + 2 * nll
    return s, b, nll, aic

def compute_ks_stat(freqs, s, b, model_type='zm'):
    n = len(freqs)
    ranks = np.arange(1, n + 1)
    
    if model_type == 'zipf':
        probs = zipf_pdf(s, n)
    else:
        probs = zipf_mandelbrot_pdf(s, b, n)
    
    cdf = np.cumsum(probs)
    ecdf = np.cumsum(freqs) / np.sum(freqs)
    ks_stat = np.max(np.abs(cdf - ecdf))
    return ks_stat

def bootstrap_ci(freqs, n_boot=100, model_type='zm'):
    n = len(freqs)
    s_boots = []
    b_boots = []
    
    # First fit to get parameters
    if model_type == 'zipf':
        s_init, _, _ = fit_zipf(freqs)
        b_init = 0.0
    else:
        s_init, b_init, _, _ = fit_zipf_mandelbrot(freqs)
    
    for i in range(n_boot):
        # Parametric bootstrap: sample from fitted model
        if model_type == 'zipf':
            probs = zipf_pdf(s_init, n)
        else:
            probs = zipf_mandelbrot_pdf(s_init, b_init, n)
        
        # Sample new frequencies from multinomial
        total = np.sum(freqs)
        boot_freqs = np.random.multinomial(total, probs)
        # Only keep non-zero frequencies (like in real data)
        boot_freqs = boot_freqs[boot_freqs > 0]
        boot_freqs = np.sort(boot_freqs)[::-1]
        
        if model_type == 'zipf':
            try:
                s_boot, _, _ = fit_zipf(boot_freqs)
                s_boots.append(s_boot)
            except:
                pass
        else:
            try:
                s_boot, b_boot, _, _ = fit_zipf_mandelbrot(boot_freqs)
                s_boots.append(s_boot)
                b_boots.append(b_boot)
            except:
                pass
    
    if len(s_boots) > 10:
        s_ci = (float(np.percentile(s_boots, 2.5)), float(np.percentile(s_boots, 97.5)))
    else:
        s_ci = None
    
    if model_type == 'zm' and len(b_boots) > 10:
        b_ci = (float(np.percentile(b_boots, 2.5)), float(np.percentile(b_boots, 97.5)))
    else:
        b_ci = None
    
    return s_ci, b_ci

def compute_r2_loglog(freqs, s, b, model_type='zm'):
    n = len(freqs)
    ranks = np.arange(1, n + 1)
    
    if model_type == 'zipf':
        probs = zipf_pdf(s, n)
    else:
        probs = zipf_mandelbrot_pdf(s, b, n)
    
    # Predicted frequencies
    total = np.sum(freqs)
    pred_freqs = probs * total
    
    # R^2 on log-log
    log_freqs = np.log(freqs)
    log_pred = np.log(np.maximum(pred_freqs, 1))
    
    ss_res = np.sum((log_freqs - log_pred) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot

def compute_residuals(freqs, s, b, model_type='zm'):
    n = len(freqs)
    ranks = np.arange(1, n + 1)
    
    if model_type == 'zipf':
        probs = zipf_pdf(s, n)
    else:
        probs = zipf_mandelbrot_pdf(s, b, n)
    
    total = np.sum(freqs)
    pred_freqs = probs * total
    
    residuals = freqs - pred_freqs
    relative_residuals = residuals / np.maximum(pred_freqs, 1)
    
    # Bin into head, body, tail
    head_end = n // 10
    body_end = n // 2
    
    bins = {
        'head': (0, head_end),
        'body': (head_end, body_end),
        'tail': (body_end, n)
    }
    
    results = {}
    for name, (start, end) in bins.items():
        if start < end:
            results[name] = {
                'mean_residual': float(np.mean(residuals[start:end])),
                'mean_relative_residual': float(np.mean(relative_residuals[start:end])),
                'std_residual': float(np.std(residuals[start:end])),
                'n': end - start
            }
    return results

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def main():
    # Load data
    data_path = Path("/Volumes/aii-s20010/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    examples = data["datasets"][0]["examples"]
    
    # Reconstruct novel
    tokens_novel = []
    examples.sort(key=lambda x: x["metadata_token_start"])
    
    for i, ex in enumerate(examples):
        text = json.loads(ex["input"])["text"]
        tokens = tokenize(text)
        step = 50
        if i < len(examples) - 1:
            tokens_novel.extend(tokens[:step])
        else:
            tokens_novel.extend(tokens)
            
    # Compute frequencies
    counts = Counter(tokens_novel)
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    
    # Fit models
    s_zipf, nll_zipf, aic_zipf = fit_zipf(freqs)
    s_zm, b_zm, nll_zm, aic_zm = fit_zipf_mandelbrot(freqs)
    
    # Goodness of fit
    ks_zipf = compute_ks_stat(freqs, s_zipf, 0, 'zipf')
    ks_zm = compute_ks_stat(freqs, s_zm, b_zm, 'zm')
    
    r2_zipf = compute_r2_loglog(freqs, s_zipf, 0, 'zipf')
    r2_zm = compute_r2_loglog(freqs, s_zm, b_zm, 'zm')
    
    # Residual analysis
    residuals_zipf = compute_residuals(freqs, s_zipf, 0, 'zipf')
    residuals_zm = compute_residuals(freqs, s_zm, b_zm, 'zm')
    
    # AIC difference
    aic_diff = aic_zipf - aic_zm
    
    # Bootstrap confidence intervals
    print("Computing bootstrap CIs...")
    np.random.seed(42)
    s_ci_zipf, _ = bootstrap_ci(freqs, n_boot=100, model_type='zipf')
    s_ci_zm, b_ci_zm = bootstrap_ci(freqs, n_boot=100, model_type='zm')
    
    # Prepare output in exp_gen_sol_out format
    output_data = {
        "metadata": {
            "method_name": "Global Zipf Fit on Pride and Prejudice",
            "description": "MLE fitting of Zipf and Zipf-Mandelbrot models to global word frequencies",
            "parameters": {
                "n_bootstrap": 100,
                "model_types": ["zipf", "zipf_mandelbrot"]
            },
            "novel_stats": {
                "total_tokens": len(tokens_novel),
                "vocabulary_size": len(freqs),
                "max_frequency": int(freqs[0]),
                "min_frequency": int(freqs[-1])
            },
            "bootstrap_ci": {
                "zipf": {"s": s_ci_zipf},
                "zipf_mandelbrot": {"s": s_ci_zm, "b": b_ci_zm}
            }
        },
        "datasets": [
            {
                "dataset": "pride_and_prejudice_global_zipf",
                "examples": [
                    {
                        "input": json.dumps({
                            "task": "fit_zipf_mle",
                            "model": "zipf",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "s": s_zipf,
                            "nll": nll_zipf,
                            "aic": aic_zipf,
                            "ks_statistic": ks_zipf,
                            "r2_loglog": r2_zipf,
                            "bootstrap_ci": s_ci_zipf
                        }),
                        "metadata_model": "zipf",
                        "metadata_n_params": 1,
                        "predict_zipf": json.dumps({"s": s_zipf, "model": "zipf"})
                    },
                    {
                        "input": json.dumps({
                            "task": "fit_zipf_mle",
                            "model": "zipf_mandelbrot",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "s": s_zm,
                            "b": b_zm,
                            "nll": nll_zm,
                            "aic": aic_zm,
                            "ks_statistic": ks_zm,
                            "r2_loglog": r2_zm,
                            "bootstrap_ci": {"s": s_ci_zm, "b": b_ci_zm}
                        }),
                        "metadata_model": "zipf_mandelbrot",
                        "metadata_n_params": 2,
                        "predict_zipf_mandelbrot": json.dumps({"s": s_zm, "b": b_zm, "model": "zipf_mandelbrot"})
                    },
                    {
                        "input": json.dumps({
                            "task": "compare_models",
                            "models": ["zipf", "zipf_mandelbrot"],
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "aic_diff_zipf_minus_zm": aic_diff,
                            "preferred_model": "zipf_mandelbrot" if aic_diff > 0 else "zipf",
                            "ks_zipf": ks_zipf,
                            "ks_zm": ks_zm,
                            "r2_zipf": r2_zipf,
                            "r2_zm": r2_zm
                        }),
                        "metadata_model": "comparison",
                        "metadata_n_params": 0,
                        "predict_comparison": json.dumps({"preferred": "zipf_mandelbrot" if aic_diff > 0 else "zipf", "aic_diff": aic_diff})
                    },
                    {
                        "input": json.dumps({
                            "task": "residual_analysis",
                            "model": "zipf",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps(residuals_zipf),
                        "metadata_model": "zipf",
                        "metadata_analysis": "residuals",
                        "predict_residuals_zipf": json.dumps({"model": "zipf", "bins": list(residuals_zipf.keys())})
                    },
                    {
                        "input": json.dumps({
                            "task": "residual_analysis",
                            "model": "zipf_mandelbrot",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps(residuals_zm),
                        "metadata_model": "zipf_mandelbrot",
                        "metadata_analysis": "residuals",
                        "predict_residuals_zm": json.dumps({"model": "zipf_mandelbrot", "bins": list(residuals_zm.keys())})
                    },
                    {
                        "input": json.dumps({
                            "task": "frequency_distribution_stats",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "n_types": len(freqs),
                            "n_tokens": int(np.sum(freqs)),
                            "hapax_legomena": int(np.sum(freqs == 1)),
                            "dis_legomena": int(np.sum(freqs == 2)),
                            "max_freq": int(freqs[0]),
                            "type_token_ratio": float(len(freqs) / np.sum(freqs)),
                            "entropy": float(-np.sum((freqs / np.sum(freqs)) * np.log(freqs / np.sum(freqs)))),
                            "freq_above_10": int(np.sum(freqs > 10)),
                            "freq_above_100": int(np.sum(freqs > 100))
                        }),
                        "metadata_analysis": "freq_stats",
                        "predict_freq_stats": json.dumps({"n_types": len(freqs), "hapax_legomena": int(np.sum(freqs == 1))})
                    },
                    {
                        "input": json.dumps({
                            "task": "per_rank_residuals",
                            "model": "zipf",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "head_top_10": {
                                "ranks": ranks[:10].tolist(),
                                "observed": freqs[:10].tolist(),
                                "predicted": (zipf_pdf(s_zipf, len(freqs)) * np.sum(freqs))[:10].tolist(),
                                "residuals": (freqs - zipf_pdf(s_zipf, len(freqs)) * np.sum(freqs))[:10].tolist()
                            }
                        }),
                        "metadata_model": "zipf",
                        "metadata_analysis": "per_rank_head",
                        "predict_head_ranks": json.dumps({"n": 10, "model": "zipf"})
                    },
                    {
                        "input": json.dumps({
                            "task": "per_rank_residuals",
                            "model": "zipf_mandelbrot",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "head_top_10": {
                                "ranks": ranks[:10].tolist(),
                                "observed": freqs[:10].tolist(),
                                "predicted": (zipf_mandelbrot_pdf(s_zm, b_zm, len(freqs)) * np.sum(freqs))[:10].tolist(),
                                "residuals": (freqs - zipf_mandelbrot_pdf(s_zm, b_zm, len(freqs)) * np.sum(freqs))[:10].tolist()
                            }
                        }),
                        "metadata_model": "zipf_mandelbrot",
                        "metadata_analysis": "per_rank_head",
                        "predict_head_ranks": json.dumps({"n": 10, "model": "zipf_mandelbrot"})
                    },
                    {
                        "input": json.dumps({
                            "task": "goodness_of_fit_by_frequency_bin",
                            "model": "zipf",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "bins": {
                                "high_freq": {
                                    "freq_range": [int(freqs[0]), int(freqs[len(freqs)//10])],
                                    "r2_loglog": float(compute_r2_loglog(freqs[:len(freqs)//10], s_zipf, 0, 'zipf')),
                                    "ks": float(compute_ks_stat(freqs[:len(freqs)//10], s_zipf, 0, 'zipf'))
                                },
                                "mid_freq": {
                                    "freq_range": [int(freqs[len(freqs)//10]), int(freqs[len(freqs)//2])],
                                    "r2_loglog": float(compute_r2_loglog(freqs[len(freqs)//10:len(freqs)//2], s_zipf, 0, 'zipf')),
                                    "ks": float(compute_ks_stat(freqs[len(freqs)//10:len(freqs)//2], s_zipf, 0, 'zipf'))
                                },
                                "low_freq": {
                                    "freq_range": [int(freqs[len(freqs)//2]), int(freqs[-1])],
                                    "r2_loglog": float(compute_r2_loglog(freqs[len(freqs)//2:], s_zipf, 0, 'zipf')),
                                    "ks": float(compute_ks_stat(freqs[len(freqs)//2:], s_zipf, 0, 'zipf'))
                                }
                            }
                        }),
                        "metadata_model": "zipf",
                        "metadata_analysis": "gof_by_bin",
                        "predict_gof_bins": json.dumps({"model": "zipf", "bins": 3})
                    },
                    {
                        "input": json.dumps({
                            "task": "goodness_of_fit_by_frequency_bin",
                            "model": "zipf_mandelbrot",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "bins": {
                                "high_freq": {
                                    "freq_range": [int(freqs[0]), int(freqs[len(freqs)//10])],
                                    "r2_loglog": float(compute_r2_loglog(freqs[:len(freqs)//10], s_zm, b_zm, 'zm')),
                                    "ks": float(compute_ks_stat(freqs[:len(freqs)//10], s_zm, b_zm, 'zm'))
                                },
                                "mid_freq": {
                                    "freq_range": [int(freqs[len(freqs)//10]), int(freqs[len(freqs)//2])],
                                    "r2_loglog": float(compute_r2_loglog(freqs[len(freqs)//10:len(freqs)//2], s_zm, b_zm, 'zm')),
                                    "ks": float(compute_ks_stat(freqs[len(freqs)//10:len(freqs)//2], s_zm, b_zm, 'zm'))
                                },
                                "low_freq": {
                                    "freq_range": [int(freqs[len(freqs)//2]), int(freqs[-1])],
                                    "r2_loglog": float(compute_r2_loglog(freqs[len(freqs)//2:], s_zm, b_zm, 'zm')),
                                    "ks": float(compute_ks_stat(freqs[len(freqs)//2:], s_zm, b_zm, 'zm'))
                                }
                            }
                        }),
                        "metadata_model": "zipf_mandelbrot",
                        "metadata_analysis": "gof_by_bin",
                        "predict_gof_bins": json.dumps({"model": "zipf_mandelbrot", "bins": 3})
                    },
                    {
                        "input": json.dumps({
                            "task": "bootstrap_parameter_stability",
                            "model": "zipf",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "n_bootstrap": 100,
                            "s_ci": s_ci_zipf,
                            "s_mean": float(np.mean([s_ci_zipf[0], s_ci_zipf[1]])) if s_ci_zipf else None,
                            "s_width": float(s_ci_zipf[1] - s_ci_zipf[0]) if s_ci_zipf else None
                        }),
                        "metadata_model": "zipf",
                        "metadata_analysis": "bootstrap_stability",
                        "predict_bootstrap_zipf": json.dumps({"param": "s", "ci": s_ci_zipf})
                    },
                    {
                        "input": json.dumps({
                            "task": "bootstrap_parameter_stability",
                            "model": "zipf_mandelbrot",
                            "data": "pride_and_prejudice_global"
                        }),
                        "output": json.dumps({
                            "n_bootstrap": 100,
                            "s_ci": s_ci_zm,
                            "b_ci": b_ci_zm,
                            "s_mean": float(np.mean([s_ci_zm[0], s_ci_zm[1]])) if s_ci_zm else None,
                            "s_width": float(s_ci_zm[1] - s_ci_zm[0]) if s_ci_zm else None,
                            "b_mean": float(np.mean([b_ci_zm[0], b_ci_zm[1]])) if b_ci_zm else None,
                            "b_width": float(b_ci_zm[1] - b_ci_zm[0]) if b_ci_zm else None
                        }),
                        "metadata_model": "zipf_mandelbrot",
                        "metadata_analysis": "bootstrap_stability",
                        "predict_bootstrap_zm": json.dumps({"params": ["s", "b"], "s_ci": s_ci_zm, "b_ci": b_ci_zm})
                    }
                ]
            }
        ]
    }
    
    with open("exp_gen_sol_out.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Also save a raw results file for easy access
    raw_results = {
        "zipf_params": {"s": float(s_zipf)},
        "zipf_mandelbrot_params": {"s": float(s_zm), "b": float(b_zm)},
        "aic_zipf": float(aic_zipf),
        "aic_zm": float(aic_zm),
        "aic_diff": float(aic_diff),
        "ks_zipf": float(ks_zipf),
        "ks_zm": float(ks_zm),
        "r2_zipf": float(r2_zipf),
        "r2_zm": float(r2_zm),
        "residuals_zipf": residuals_zipf,
        "residuals_zm": residuals_zm,
        "bootstrap_ci": {
            "zipf": {"s": s_ci_zipf},
            "zipf_mandelbrot": {"s": s_ci_zm, "b": b_ci_zm}
        },
        "novel_stats": {
            "total_tokens": len(tokens_novel),
            "vocabulary_size": len(freqs)
        }
    }
    
    with open("results.json", "w") as f:
        json.dump(raw_results, f, indent=2)

if __name__ == "__main__":
    main()
