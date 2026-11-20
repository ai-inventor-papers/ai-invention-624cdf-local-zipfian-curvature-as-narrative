#!/usr/bin/env python3
"""
Independent audit script to re-derive headline numbers from raw data.
This uses a completely different code path than method.py.
"""

import json
import re
import numpy as np
from collections import Counter
from pathlib import Path
from scipy.optimize import minimize

# Load raw data from dependency
data_path = Path("/Volumes/aii-s20010/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json")
with open(data_path, "r") as f:
    data = json.load(f)

examples = data["datasets"][0]["examples"]

def tokenize(text):
    """Use SAME tokenization as method.py for fair comparison"""
    return re.findall(r"\b\w+\b", text.lower())

# Reconstruct novel using SAME approach as method.py: take first 50 tokens from each window (step size)
examples.sort(key=lambda x: x["metadata_token_start"])
tokens_audit = []

for i, ex in enumerate(examples):
    text = json.loads(ex["input"])["text"]
    tokens = tokenize(text)
    step = 50
    if i < len(examples) - 1:
        tokens_audit.extend(tokens[:step])
    else:
        tokens_audit.extend(tokens)

print(f"Total tokens (audit): {len(tokens_audit)}")

# Count frequencies
counts = Counter(tokens_audit)
freqs = np.array(sorted(counts.values(), reverse=True))
n = len(freqs)
ranks = np.arange(1, n + 1)
total = np.sum(freqs)

print(f"Vocabulary size (audit): {n}")
print(f"Max frequency: {freqs[0]}")
print(f"Min frequency: {freqs[-1]}")

# --- Independent MLE fitting using different optimization approach ---
# Use negative log-likelihood with explicit multinomial formulation

def negloglik_zipf(s, freqs):
    """Negative log-likelihood for Zipf: p(r) = C / r^s"""
    r = np.arange(1, len(freqs) + 1)
    log_probs = -s * np.log(r)
    # Normalize: log C = -log(sum(r^-s))
    log_C = -np.logaddexp.reduce(log_probs)
    log_probs += log_C
    return -np.sum(freqs * log_probs)

def negloglik_zm(params, freqs):
    """Negative log-likelihood for Zipf-Mandelbrot: p(r) = C / (r+b)^s"""
    s, b = params
    r = np.arange(1, len(freqs) + 1)
    log_probs = -s * np.log(r + b)
    log_C = -np.logaddexp.reduce(log_probs)
    log_probs += log_C
    return -np.sum(freqs * log_probs)

# Fit Zipf using scalar minimize
print("\n=== Fitting Zipf (audit) ===")
res_zipf = minimize(negloglik_zipf, x0=[1.0], args=(freqs,), 
                    bounds=[(0.01, 5.0)], method='L-BFGS-B')
s_zipf_audit = res_zipf.x[0]
nll_zipf_audit = res_zipf.fun
aic_zipf_audit = 2 * 1 + 2 * nll_zipf_audit
print(f"s = {s_zipf_audit:.6f}")
print(f"NLL = {nll_zipf_audit:.6f}")
print(f"AIC = {aic_zipf_audit:.6f}")

# Fit Zipf-Mandelbrot
print("\n=== Fitting Zipf-Mandelbrot (audit) ===")
res_zm = minimize(negloglik_zm, x0=[1.0, 1.0], args=(freqs,),
                  bounds=[(0.01, 5.0), (0.0, 100.0)], method='L-BFGS-B')
s_zm_audit, b_zm_audit = res_zm.x[0], res_zm.x[1]
nll_zm_audit = res_zm.fun
aic_zm_audit = 2 * 2 + 2 * nll_zm_audit
print(f"s = {s_zm_audit:.6f}")
print(f"b = {b_zm_audit:.6f}")
print(f"NLL = {nll_zm_audit:.6f}")
print(f"AIC = {aic_zm_audit:.6f}")

# --- Compute KS statistics independently ---
def compute_ks(freqs, s, b=0, model='zipf'):
    n = len(freqs)
    r = np.arange(1, n + 1)
    if model == 'zipf':
        log_probs = -s * np.log(r)
    else:
        log_probs = -s * np.log(r + b)
    log_C = -np.logaddexp.reduce(log_probs)
    probs = np.exp(log_probs + log_C)
    cdf = np.cumsum(probs)
    ecdf = np.cumsum(freqs) / np.sum(freqs)
    return np.max(np.abs(cdf - ecdf))

ks_zipf_audit = compute_ks(freqs, s_zipf_audit, model='zipf')
ks_zm_audit = compute_ks(freqs, s_zm_audit, b_zm_audit, model='zm')
print(f"\nKS Zipf: {ks_zipf_audit:.6f}")
print(f"KS ZM: {ks_zm_audit:.6f}")

# --- Compute R^2 on log-log independently ---
def compute_r2_loglog(freqs, s, b=0, model='zipf'):
    n = len(freqs)
    r = np.arange(1, n + 1)
    if model == 'zipf':
        log_probs = -s * np.log(r)
    else:
        log_probs = -s * np.log(r + b)
    log_C = -np.logaddexp.reduce(log_probs)
    probs = np.exp(log_probs + log_C)
    pred = probs * np.sum(freqs)
    
    log_obs = np.log(freqs)
    log_pred = np.log(np.maximum(pred, 1))
    
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    
    return 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

r2_zipf_audit = compute_r2_loglog(freqs, s_zipf_audit, model='zipf')
r2_zm_audit = compute_r2_loglog(freqs, s_zm_audit, b_zm_audit, model='zm')
print(f"\nR2 Zipf: {r2_zipf_audit:.6f}")
print(f"R2 ZM: {r2_zm_audit:.6f}")

# --- Compare with original results ---
print("\n=== COMPARISON WITH ORIGINAL RESULTS ===")
with open("results.json", "r") as f:
    orig = json.load(f)

print(f"s_zipf:  original={orig['zipf_params']['s']:.6f}, audit={s_zipf_audit:.6f}, diff={abs(orig['zipf_params']['s'] - s_zipf_audit):.6f}")
print(f"s_zm:    original={orig['zipf_mandelbrot_params']['s']:.6f}, audit={s_zm_audit:.6f}, diff={abs(orig['zipf_mandelbrot_params']['s'] - s_zm_audit):.6f}")
print(f"b_zm:    original={orig['zipf_mandelbrot_params']['b']:.6f}, audit={b_zm_audit:.6f}, diff={abs(orig['zipf_mandelbrot_params']['b'] - b_zm_audit):.6f}")
print(f"KS_zipf: original={orig['ks_zipf']:.6f}, audit={ks_zipf_audit:.6f}, diff={abs(orig['ks_zipf'] - ks_zipf_audit):.6f}")
print(f"KS_zm:   original={orig['ks_zm']:.6f}, audit={ks_zm_audit:.6f}, diff={abs(orig['ks_zm'] - ks_zm_audit):.6f}")
print(f"R2_zipf: original={orig['r2_zipf']:.6f}, audit={r2_zipf_audit:.6f}, diff={abs(orig['r2_zipf'] - r2_zipf_audit):.6f}")
print(f"R2_zm:   original={orig['r2_zm']:.6f}, audit={r2_zm_audit:.6f}, diff={abs(orig['r2_zm'] - r2_zm_audit):.6f}")
print(f"AIC_zipf: original={orig['aic_zipf']:.2f}, audit={aic_zipf_audit:.2f}, diff={abs(orig['aic_zipf'] - aic_zipf_audit):.2f}")
print(f"AIC_zm:   original={orig['aic_zm']:.2f}, audit={aic_zm_audit:.2f}, diff={abs(orig['aic_zm'] - aic_zm_audit):.2f}")

# Tolerance check
tol = 1e-4
all_match = True
for name, orig_val, audit_val in [
    ("s_zipf", orig['zipf_params']['s'], s_zipf_audit),
    ("s_zm", orig['zipf_mandelbrot_params']['s'], s_zm_audit),
    ("b_zm", orig['zipf_mandelbrot_params']['b'], b_zm_audit),
    ("KS_zipf", orig['ks_zipf'], ks_zipf_audit),
    ("KS_zm", orig['ks_zm'], ks_zm_audit),
    ("R2_zipf", orig['r2_zipf'], r2_zipf_audit),
    ("R2_zm", orig['r2_zm'], r2_zm_audit),
    ("AIC_zipf", orig['aic_zipf'], aic_zipf_audit),
    ("AIC_zm", orig['aic_zm'], aic_zm_audit),
]:
    if abs(orig_val - audit_val) > tol:
        print(f"MISMATCH: {name} diff = {abs(orig_val - audit_val):.6f} > {tol}")
        all_match = False

if all_match:
    print("\n✓ ALL HEADLINE NUMBERS MATCH within tolerance!")
else:
    print("\n✗ SOME NUMBERS DO NOT MATCH")

# --- Sanity check: test on random/shuffled data ---
print("\n=== SANITY CHECK: SHUFFLED FREQUENCIES ===")
np.random.seed(123)
shuffled = np.random.permutation(freqs)
# Shuffled data should NOT fit Zipf well
# Fit Zipf to shuffled
res_shuf = minimize(negloglik_zipf, x0=[1.0], args=(shuffled,), 
                    bounds=[(0.01, 5.0)], method='L-BFGS-B')
s_shuf = res_shuf.x[0]
ks_shuf = compute_ks(shuffled, s_shuf, model='zipf')
r2_shuf = compute_r2_loglog(shuffled, s_shuf, model='zipf')
print(f"Shuffled data - s: {s_shuf:.4f}, KS: {ks_shuf:.4f}, R2: {r2_shuf:.4f}")
print(f"Original data - s: {s_zipf_audit:.4f}, KS: {ks_zipf_audit:.4f}, R2: {r2_zipf_audit:.4f}")

if ks_shuf > ks_zipf_audit and r2_shuf < r2_zipf_audit:
    print("✓ SANITY CHECK PASSED: Shuffled data fits worse (higher KS, lower R2)")
else:
    print("✗ SANITY CHECK FAILED: Shuffled data fits similarly or better")