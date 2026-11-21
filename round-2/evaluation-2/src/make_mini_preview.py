import json

def truncate_value(value, max_len=200):
    if isinstance(value, list):
        return [truncate_value(item, max_len) for item in value[:3]]
    if isinstance(value, str):
        if len(value) > max_len:
            return value[:max_len] + "..."
        return value
    if isinstance(value, dict):
        return {key: truncate_value(val, max_len) for key, val in value.items()}
    return value

with open('/Volumes/aii-s20003/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_2/gen_art/gen_art_evaluation_2/exp_eval_sol_out.json') as f:
    data = json.load(f)

# Full version
full_data = data
with open('/Volumes/aii-s20003/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_2/gen_art/gen_art_evaluation_2/full_exp_eval_sol_out.json', 'w') as f:
    json.dump(full_data, f, indent=2)

# Mini version - slice examples
mini_data = {**data}
mini_data['datasets'] = []
for ds in data['datasets']:
    mini_ds = {**ds}
    mini_ds['examples'] = ds['examples'][:3]
    mini_data['datasets'].append(mini_ds)

with open('/Volumes/aii-s20003/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_2/gen_art/gen_art_evaluation_2/mini_exp_eval_sol_out.json', 'w') as f:
    json.dump(mini_data, f, indent=2)

# Preview version - truncate strings
preview_data = truncate_value(data)
with open('/Volumes/aii-s20003/work/run_fODCbnzS2ZPQ/3_invention_loop/iter_2/gen_art/gen_art_evaluation_2/preview_exp_eval_sol_out.json', 'w') as f:
    json.dump(preview_data, f, indent=2)

print("Generated full, mini, preview versions")