import json
import numpy as np
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, "/Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts")
from chart_gen import render

# Load results
with open("results.json", "r") as f:
    results = json.load(f)

# Reconstruct frequencies
data_path = Path("/Volumes/aii-s20010/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json")
with open(data_path, "r") as f:
    data = json.load(f)

examples = data["datasets"][0]["examples"]

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

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

counts = Counter(tokens_novel)
freqs = np.array(sorted(counts.values(), reverse=True))
n = len(freqs)
ranks = np.arange(1, n + 1)

# Model parameters
s_zipf = results["zipf_params"]["s"]
s_zm = results["zipf_mandelbrot_params"]["s"]
b_zm = results["zipf_mandelbrot_params"]["b"]

# Compute predicted frequencies
def zipf_pdf(s, n):
    r_arr = np.arange(1, n + 1)
    probs = 1.0 / np.power(r_arr, s)
    return probs / np.sum(probs)

def zipf_mandelbrot_pdf(s, b, n):
    r_arr = np.arange(1, n + 1)
    probs = 1.0 / np.power(r_arr + b, s)
    return probs / np.sum(probs)

total = np.sum(freqs)
zipf_probs = zipf_pdf(s_zipf, n)
zm_probs = zipf_mandelbrot_pdf(s_zm, b_zm, n)
zipf_pred = zipf_probs * total
zm_pred = zm_probs * total

# ===========================================
# Figure 1: Log-log rank-frequency plot with both fits
# ===========================================
fig1_spec = {
    "type": "line",
    "title": "Zipf and Zipf-Mandelbrot fits to Pride and Prejudice word frequencies",
    "xlabel": "Rank",
    "ylabel": "Frequency",
    "aspect": "4:3",
    "logx": True,
    "logy": True,
    "series": [
        {
            "label": "Empirical",
            "x": ranks.tolist(),
            "values": freqs.tolist(),
            "marker": "o",
            "markersize": 3,
            "linestyle": "none"
        },
        {
            "label": f"Zipf (s={s_zipf:.3f})",
            "x": ranks.tolist(),
            "values": zipf_pred.tolist(),
            "marker": "none",
            "linestyle": "--"
        },
        {
            "label": f"Zipf-Mandelbrot (s={s_zm:.3f}, b={b_zm:.1f})",
            "x": ranks.tolist(),
            "values": zm_pred.tolist(),
            "marker": "none",
            "linestyle": "-"
        }
    ]
}

# ===========================================
# Figure 2: Residuals across rank bins
# ===========================================
# Compute residuals per bin
head_end = n // 10
body_end = n // 2

bins = {
    'head': (0, head_end),
    'body': (head_end, body_end),
    'tail': (body_end, n)
}

zipf_residuals_data = []
zm_residuals_data = []

for name, (start, end) in bins.items():
    if start < end:
        zipf_res = freqs[start:end] - zipf_pred[start:end]
        zm_res = freqs[start:end] - zm_pred[start:end]
        zipf_residuals_data.append(float(np.mean(zipf_res)))
        zm_residuals_data.append(float(np.mean(zm_res)))

fig2_spec = {
    "type": "bar",
    "title": "Mean residuals by frequency rank region",
    "xlabel": "Region",
    "ylabel": "Mean residual (observed - predicted)",
    "aspect": "4:3",
    "categories": ["Head (top 10%)", "Body (10-50%)", "Tail (bottom 50%)"],
    "series": [
        {
            "label": "Zipf",
            "values": zipf_residuals_data
        },
        {
            "label": "Zipf-Mandelbrot",
            "values": zm_residuals_data
        }
    ]
}

# ===========================================
# Figure 3: Goodness-of-fit comparison
# ===========================================
fig3_spec = {
    "type": "bar",
    "title": "Goodness-of-fit comparison: Zipf vs Zipf-Mandelbrot",
    "xlabel": "Metric",
    "ylabel": "Value",
    "aspect": "4:3",
    "categories": ["KS statistic", "R² (log-log)", "AIC"],
    "series": [
        {
            "label": "Zipf",
            "values": [
                results["ks_zipf"],
                results["r2_zipf"],
                results["aic_zipf"]
            ]
        },
        {
            "label": "Zipf-Mandelbrot",
            "values": [
                results["ks_zm"],
                results["r2_zm"],
                results["aic_zm"]
            ]
        }
    ]
}

# ===========================================
# Figure 4: Bootstrap confidence intervals
# ===========================================
fig4_spec = {
    "type": "forest",
    "title": "Bootstrap 95% confidence intervals for model parameters",
    "xlabel": "Parameter value",
    "aspect": "4:3",
    "items": [
        {
            "label": "Zipf: s",
            "estimate": s_zipf,
            "ci": results["bootstrap_ci"]["zipf"]["s"]
        },
        {
            "label": "Zipf-Mandelbrot: s",
            "estimate": s_zm,
            "ci": results["bootstrap_ci"]["zipf_mandelbrot"]["s"]
        },
        {
            "label": "Zipf-Mandelbrot: b",
            "estimate": b_zm,
            "ci": results["bootstrap_ci"]["zipf_mandelbrot"]["b"]
        }
    ],
    "null_line": None
}

# Render figures
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

print("Rendering Figure 1: Log-log rank-frequency plot...")
render(fig1_spec, out=figures_dir / "fig1_rank_freq")

print("Rendering Figure 2: Residuals by rank region...")
render(fig2_spec, out=figures_dir / "fig2_residuals")

print("Rendering Figure 3: Goodness-of-fit comparison...")
render(fig3_spec, out=figures_dir / "fig3_gof")

print("Rendering Figure 4: Bootstrap CIs...")
render(fig4_spec, out=figures_dir / "fig4_bootstrap_ci")

print("All figures rendered!")