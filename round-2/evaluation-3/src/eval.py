import json
import numpy as np
from scipy.optimize import minimize
import collections
import re

def neg_log_likelihood(params, ranks, frequencies):
    s, b, C = params
    lambda_r = C * (ranks + b)**(-s)
    lambda_r = np.maximum(lambda_r, 1e-10)
    return np.sum(lambda_r - frequencies * np.log(lambda_r))

def compute_ks_statistic(ranks, frequencies, params):
    s, b, C = params
    lambda_r = C * (ranks + b)**(-s)
    cdf_model = np.cumsum(lambda_r) / np.sum(lambda_r)
    cdf_empirical = np.cumsum(frequencies) / np.sum(frequencies)
    return np.max(np.abs(cdf_model - cdf_empirical))

def compute_aic_bic(n, k, log_likelihood):
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def recompute_statistics(data):
    examples = data['datasets'][0]['examples']
    total_windows = len(examples)
    
    unique_tokens = []
    function_word_counts = []
    type_token_ratios = []
    function_word_ratios = []
    content_word_counts = []
    total_tokens_list = []
    
    for ex in examples:
        out = json.loads(ex['output'])
        unique_tokens.append(out['unique_tokens'])
        function_word_counts.append(out['function_word_count'])
        type_token_ratios.append(out['type_token_ratio'])
        function_word_ratios.append(out['function_word_ratio'])
        content_word_counts.append(out['content_word_count'])
        total_tokens_list.append(out['total_tokens'])
    
    stats = {
        'total_windows': total_windows,
        'mean_unique_tokens_per_window': float(np.mean(unique_tokens)),
        'mean_function_word_count': float(np.mean(function_word_counts)),
        'mean_function_word_ratio': float(np.mean(function_word_ratios)),
        'mean_type_token_ratio': float(np.mean(type_token_ratios)),
        'mean_content_word_count': float(np.mean(content_word_counts)),
        'mean_total_tokens': float(np.mean(total_tokens_list)),
    }
    return stats

def main():
    # Load full novel text
    with open('pride_and_prejudice_clean.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = re.findall(r'\b\w+\b', text.lower())
    counts = collections.Counter(tokens)
    frequencies = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(frequencies) + 1)
    
    n = len(frequencies)
    
    # 1. Recompute statistics from windowed data
    data = load_data('full_data_out.json')
    stats = recompute_statistics(data)
    
    print("=== Recomputed Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Original stats (from the artifact plan)
    original_stats = {
        'total_windows': 2453,
        'mean_unique_tokens_per_window': 750,
        'mean_function_word_count': None,  # Not given in plan
        'mean_function_word_ratio': None,
        'mean_type_token_ratio': None,
    }
    
    # 2. Fit Zipf-Mandelbrot (3 parameters: s, b, C)
    initial_params = [1.0, 1.0, frequencies[0]]
    result = minimize(neg_log_likelihood, initial_params, args=(ranks, frequencies), method='L-BFGS-B', bounds=[(0.1, 5), (0, 100), (0, None)])
    s_zm, b_zm, C_zm = result.x
    nll_zm = result.fun
    log_likelihood_zm = -nll_zm
    aic_zm, bic_zm = compute_aic_bic(n, 3, log_likelihood_zm)
    ks_zm = compute_ks_statistic(ranks, frequencies, [s_zm, b_zm, C_zm])
    
    print(f"\n=== Zipf-Mandelbrot ===")
    print(f"  s (exponent): {s_zm:.4f}")
    print(f"  b (curvature): {b_zm:.4f}")
    print(f"  C (normalization): {C_zm:.2f}")
    print(f"  Log-likelihood: {log_likelihood_zm:.2f}")
    print(f"  AIC: {aic_zm:.2f}")
    print(f"  BIC: {bic_zm:.2f}")
    print(f"  KS Statistic: {ks_zm:.4f}")
    
    # 3. Fit Zipf (Power Law) - fix b=0 (2 parameters: s, C)
    def neg_log_likelihood_zipf(params, ranks, frequencies):
        s, C = params
        b = 0.0
        lambda_r = C * (ranks + b)**(-s)
        lambda_r = np.maximum(lambda_r, 1e-10)
        return np.sum(lambda_r - frequencies * np.log(lambda_r))
    
    initial_params_zipf = [1.0, frequencies[0]]
    result_zipf = minimize(neg_log_likelihood_zipf, initial_params_zipf, args=(ranks, frequencies), method='L-BFGS-B', bounds=[(0.1, 5), (0, None)])
    s_zipf, C_zipf = result_zipf.x
    nll_zipf = result_zipf.fun
    log_likelihood_zipf = -nll_zipf
    aic_zipf, bic_zipf = compute_aic_bic(n, 2, log_likelihood_zipf)
    ks_zipf = compute_ks_statistic(ranks, frequencies, [s_zipf, 0.0, C_zipf])
    
    print(f"\n=== Zipf (Power Law) ===")
    print(f"  s (exponent): {s_zipf:.4f}")
    print(f"  C (normalization): {C_zipf:.2f}")
    print(f"  Log-likelihood: {log_likelihood_zipf:.2f}")
    print(f"  AIC: {aic_zipf:.2f}")
    print(f"  BIC: {bic_zipf:.2f}")
    print(f"  KS Statistic: {ks_zipf:.4f}")
    
    # 4. Generate log-log plot data
    # Save model predictions for plotting
    model_zm = C_zm * (ranks + b_zm)**(-s_zm)
    model_zipf = C_zipf * ranks**(-s_zipf)
    
    # 5. Prepare results for output
    zipf_results = {
        'model': 'power_law',
        's': float(s_zipf),
        'b': 0.0,
        'C': float(C_zipf),
        'log_likelihood': float(log_likelihood_zipf),
        'aic': float(aic_zipf),
        'bic': float(bic_zipf),
        'ks_statistic': float(ks_zipf),
        'num_parameters': 2
    }
    
    zm_results = {
        'model': 'zipf_mandelbrot',
        's': float(s_zm),
        'b': float(b_zm),
        'C': float(C_zm),
        'log_likelihood': float(log_likelihood_zm),
        'aic': float(aic_zm),
        'bic': float(bic_zm),
        'ks_statistic': float(ks_zm),
        'num_parameters': 3
    }
    
    # 6. Stats comparison
    stats_comparison = {
        'original': original_stats,
        'corrected': stats,
        'differences': {}
    }
    for k in original_stats:
        if k in stats and original_stats[k] is not None:
            stats_comparison['differences'][k] = float(stats[k] - original_stats[k])
    
    # Save results in exp_eval_sol_out format
    # Create metrics_agg with key metrics
    metrics_agg = {
        'zipf_exponent_s': s_zipf,
        'zipf_mandelbrot_exponent_s': s_zm,
        'zipf_mandelbrot_curvature_b': b_zm,
        'zipf_log_likelihood': log_likelihood_zipf,
        'zipf_mandelbrot_log_likelihood': log_likelihood_zm,
        'zipf_aic': aic_zipf,
        'zipf_mandelbrot_aic': aic_zm,
        'zipf_bic': bic_zipf,
        'zipf_mandelbrot_bic': bic_zm,
        'zipf_ks_statistic': ks_zipf,
        'zipf_mandelbrot_ks_statistic': ks_zm,
        'mean_unique_tokens_per_window': stats['mean_unique_tokens_per_window'],
        'mean_function_word_ratio': stats['mean_function_word_ratio'],
        'mean_type_token_ratio': stats['mean_type_token_ratio'],
        'total_windows': stats['total_windows']
    }
    
    # Create examples for the dataset
    # Each example represents a model comparison
    examples = [
        {
            "input": "Evaluate Zipf (Power Law) fit on Pride and Prejudice word frequency distribution",
            "output": json.dumps(zipf_results),
            "metadata_model": "power_law",
            "metadata_num_parameters": 2,
            "eval_log_likelihood": log_likelihood_zipf,
            "eval_aic": aic_zipf,
            "eval_bic": bic_zipf,
            "eval_ks_statistic": ks_zipf,
            "predict_s": str(s_zipf),
            "predict_b": "0.0",
            "predict_C": str(C_zipf)
        },
        {
            "input": "Evaluate Zipf-Mandelbrot fit on Pride and Prejudice word frequency distribution",
            "output": json.dumps(zm_results),
            "metadata_model": "zipf_mandelbrot",
            "metadata_num_parameters": 3,
            "eval_log_likelihood": log_likelihood_zm,
            "eval_aic": aic_zm,
            "eval_bic": bic_zm,
            "eval_ks_statistic": ks_zm,
            "predict_s": str(s_zm),
            "predict_b": str(b_zm),
            "predict_C": str(C_zm)
        }
    ]
    
    output = {
        "metadata": {
            "evaluation_name": "Global Zipfian Fit and Statistics Correction",
            "description": "MLE fitting of Zipf and Zipf-Mandelbrot models to Pride and Prejudice word frequencies with statistics correction",
            "parameters": {
                "text_source": "Pride and Prejudice (Project Gutenberg)",
                "method": "MLE with Poisson likelihood",
                "novel_tokens": len(tokens),
                "vocabulary_size": len(counts)
            }
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "pride_and_prejudice_zipf_fit",
                "examples": examples
            }
        ]
    }
    
    with open('eval_out.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save stats comparison separately
    with open('stats_comparison.json', 'w') as f:
        json.dump(stats_comparison, f, indent=2)
    
    # Generate diff.md
    with open('diff.md', 'w') as f:
        f.write("# Statistics Comparison: Original vs. Corrected\n\n")
        f.write("| Statistic | Original | Corrected | Difference |\n")
        f.write("|-----------|----------|-----------|------------|\n")
        for k in stats:
            orig = original_stats.get(k, 'N/A')
            corr = stats[k]
            if orig is not None and orig != 'N/A':
                diff = corr - orig
                f.write(f"| {k} | {orig} | {corr:.4f} | {diff:.4f} |\n")
            else:
                f.write(f"| {k} | N/A | {corr:.4f} | N/A |\n")
        f.write("\n## Notes\n")
        f.write("- The original claim of 'mean unique tokens per window: 750' appears to be incorrect. ")
        f.write("The window size is 750 tokens, but unique tokens per window is much lower (~337).\n")
        f.write("- Function word ratio and type-token ratio were not provided in original stats.\n")
    
    print("\nResults saved to eval_out.json, stats_comparison.json, and diff.md")

if __name__ == '__main__':
    main()