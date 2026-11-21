
import numpy as np
import scipy.optimize as opt
from collections import Counter
import re

def get_word_frequencies(text):
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    frequencies = sorted(counts.values(), reverse=True)
    return np.array(frequencies)

def zipf_mandelbrot_prob(r, s, b):
    # P(r) = 1 / ((r+b)^s * sum_{j=1}^N (j+b)^-s)
    # This needs normalization, let's use the raw form and normalize
    # Actually, for MLE, we use the probability mass function:
    # P(r) = (r+b)^-s / sum_{k=1}^R (k+b)^-s
    pass

# Simplified: MLE fitting for power-law
# Reference: Clauset et al. 2009
# For Zipf: P(r) = C * r^-s
# For Zipf-Mandelbrot: P(r) = C * (r+b)^-s

# Testing loading
with open('pride_and_prejudice_clean.txt', 'r') as f:
    text = f.read()

freqs = get_word_frequencies(text)
print(f"Number of unique words: {len(freqs)}")
print(f"Top 10 frequencies: {freqs[:10]}")
