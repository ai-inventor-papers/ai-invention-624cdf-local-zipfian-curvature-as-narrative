import json
import numpy as np
import re
from collections import Counter
from pathlib import Path
import subprocess

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
# Downsample for readability (every 10th point)
step = max(1, n // 2000)
ranks_sub = ranks[::step].tolist()
freqs_sub = freqs[::step].tolist()
zipf_pred_sub = zipf_pred[::step].tolist()
zm_pred_sub = zm_pred[::step].tolist()

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
            "x": ranks_sub,
            "values": freqs_sub,
            "marker": "o"
        },
        {
            "label": f"Zipf (s={s_zipf:.3f})",
            "x": ranks_sub,
            "values": zipf_pred_sub,
            "marker": "none",
            "linestyle": "--"
        },
        {
            "label": f"Zipf-Mandelbrot (s={s_zm:.3f}, b={b_zm:.1f})",
            "x": ranks_sub,
            "values": zm_pred_sub,
            "marker": "none",
            "linestyle": "-"
        }
    ]
}

# ===========================================
# Figure 2: Residuals across rank bins
# ===========================================
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
# Convert CI to errors (half-width)
ci_zipf_s = results["bootstrap_ci"]["zipf"]["s"]
ci_zm_s = results["bootstrap_ci"]["zipf_mandelbrot"]["s"]
ci_zm_b = results["bootstrap_ci"]["zipf_mandelbrot"]["b"]

zipf_s_err = (ci_zipf_s[1] - ci_zipf_s[0]) / 2
zm_s_err = (ci_zm_s[1] - ci_zm_s[0]) / 2
zm_b_err = (ci_zm_b[1] - ci_zm_b[0]) / 2

fig4_spec = {
    "type": "forest",
    "title": "Bootstrap 95% confidence intervals for model parameters",
    "xlabel": "Parameter value",
    "aspect": "4:3",
    "categories": [
        "Zipf: s",
        "Zipf-Mandelbrot: s",
        "Zipf-Mandelbrot: b"
    ],
    "series": [
        {
            "values": [float(s_zipf), float(s_zm), float(b_zm)],
            "errors": [float(zipf_s_err), float(zm_s_err), float(zm_b_err)]
        }
    ],
    "null_line": 0.0
}

# Write specs and render
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

specs = {
    "fig1_rank_freq": fig1_spec,
    "fig2_residuals": fig2_spec,
    "fig3_gof": fig3_spec,
    "fig4_bootstrap_ci": fig4_spec
}

for name, spec in specs.items():
    spec_path = figures_dir / f"{name}_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    
    out_path = figures_dir / name
    print(f"Rendering {name}...")
    result = subprocess.run(
        ["python", "/Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py", 
         "--spec", str(spec_path), "--out", str(out_path)],
        capture_output=True, text=True, cwd="/Volumes/aii-s20010/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_2/gen_art/gen_art_experiment_2"
    )
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
    else:
        print(f"  Done")

print("All figures rendered!")