import json
import numpy as np

# Load the evaluation results
with open('eval_out.json', 'r') as f:
    data = json.load(f)

ranks = data['raw_data']['ranks']
frequencies = data['raw_data']['frequencies']
model_zipf = data['raw_data']['model_zipf']
model_zm = data['raw_data']['model_zm']

# Create a subset for plotting (log-spaced sampling for better visualization)
n = len(ranks)
# Sample points: first 100, then log-spaced
indices = list(range(0, min(100, n)))
if n > 100:
    # Add log-spaced points
    log_indices = np.logspace(np.log10(100), np.log10(n-1), 100, dtype=int)
    indices.extend(log_indices)

# Remove duplicates
indices = sorted(set(indices))

# Create the spec using line type (which supports x values)
spec = {
    "type": "line",
    "title": "Zipfian frequency distribution of Pride and Prejudice",
    "xlabel": "Rank",
    "ylabel": "Frequency",
    "aspect": "4:3",
    "series": [
        {
            "label": "Empirical",
            "x": [ranks[i] for i in indices],
            "values": [frequencies[i] for i in indices]
        },
        {
            "label": "Zipf (Power Law)",
            "x": [ranks[i] for i in indices],
            "values": [model_zipf[i] for i in indices]
        },
        {
            "label": "Zipf-Mandelbrot",
            "x": [ranks[i] for i in indices],
            "values": [model_zm[i] for i in indices]
        }
    ]
}

with open('zipf_spec.json', 'w') as f:
    json.dump(spec, f, indent=2)

print(f"Created spec with {len(indices)} points")
print(f"Rank range: {min(ranks[i] for i in indices)} to {max(ranks[i] for i in indices)}")