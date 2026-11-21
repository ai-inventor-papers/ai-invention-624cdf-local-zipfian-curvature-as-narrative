# Reproducibility Instructions

This document provides step-by-step instructions to reproduce the Rolling Zipfian Narrative Windows dataset from Pride and Prejudice.

## 1. Get the Artifact

Clone the repository and navigate to this artifact's folder:

```bash
git clone <repository-url>
cd <repository-root>/round-1/dataset-1/src
```

## 2. System Requirements

- **OS**: Ubuntu 22.04+ (or macOS with Homebrew)
- **Python**: 3.12+
- **System packages**: `build-essential`, `python3-dev`, `libssl-dev`

```bash
# Ubuntu
sudo apt-get update && sudo apt-get install -y build-essential python3-dev libssl-dev

# macOS (using Homebrew)
brew install python@3.12 openssl
```

## 3. Environment Setup

Create and activate a virtual environment, then install exact dependency versions:

```bash
# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install exact versions from pyproject.toml
pip install -r pyproject.toml
```

Or manually install the pinned versions:

```bash
pip install \
    aii-pipeline==0+bundle \
    attrs==26.1.0 \
    certifi==2026.10.23 \
    click==8.5.1 \
    cloudpickle==3.1.3 \
    defusedxml==0.7.1 \
    datasets==5.2.0 \
    dill==0.4.1 \
    filelock==4.2.12 \
    fsspec==2026.9.0 \
    h11==0.16.0 \
    hf-xet==1.8.0 \
    httpcore==1.0.9 \
    httpcore2==2.16.2 \
    httpx==0.28.1 \
    httpx2==2.16.2 \
    huggingface-hub==2.4.0 \
    idna==3.20 \
    joblib==1.6.0 \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    multiprocess==0.70.19 \
    nltk==3.10.4 \
    numpy==2.5.5 \
    packaging==26.3 \
    pandas==3.1.1 \
    pyarrow==26.0.1 \
    python-dateutil==4.0.0 \
    python-dotenv==1.2.4 \
    PyYAML==6.0.3 \
    referencing==0.37.0 \
    regex==2026.11.19 \
    rpds-py==2026.9.1 \
    tqdm==4.70.1 \
    truststore==0.10.5 \
    tzdata==2026.6 \
    typing-extensions==4.16.0 \
    xxhash==4.1.0 \
    anyio==4.16.0 \
    loguru==0.8.0
```

## 4. Download NLTK Stopwords

```bash
python -c "import nltk; nltk.download('stopwords', quiet=True)"
```

## 5. Run the Dataset Generation

```bash
# Generate the dataset (uses already downloaded novel text)
python data.py
```

**Approximate runtime**: ~10 seconds on a modern CPU (no GPU needed)

## 6. Expected Outputs

After successful execution, you should have:

| File | Size | Description |
|------|------|-------------|
| `full_data_out.json` | ~13 MB | Complete dataset with 2,453 windows |
| `mini_data_out.json` | ~15 KB | First 3 examples for testing |
| `preview_data_out.json` | ~3 KB | First 3 examples with truncated strings |
| `pride_and_prejudice_novel.txt` | ~675 KB | Clean novel text |
| `logs/dataset_generation.log` | ~few KB | Execution log |

## 7. Verify the Output

Validate against the schema:

```bash
python -c "
import json
from jsonschema import validate

with open('/path/to/aii-json/schemas/exp_sel_data_out.json') as f:
    schema = json.load(f)
with open('full_data_out.json') as f:
    data = json.load(f)
validate(instance=data, schema=schema)
print('Validation passed!')
"
```

Check dataset structure:

```bash
python -c "
import json
with open('full_data_out.json') as f:
    data = json.load(f)
ds = data['datasets'][0]
print(f'Dataset: {ds[\"dataset\"]}')
print(f'Examples: {len(ds[\"examples\"])}')
ex = ds['examples'][0]
print(f'Input keys: {list(json.loads(ex[\"input\"]).keys())}')
print(f'Output keys: {list(json.loads(ex[\"output\"]).keys())}')
print(f'Metadata keys: {[k for k in ex if k.startswith(\"metadata_\")]}')
"
```

Expected output:
```
Dataset: pride_and_prejudice_rolling_windows
Examples: 2453
Input keys: ['text', 'token_start', 'token_end', 'char_start', 'char_end']
Output keys: ['total_tokens', 'unique_tokens', 'function_word_count', 'content_word_count', 'function_word_ratio', 'type_token_ratio']
Metadata keys: ['metadata_window_index', 'metadata_token_start', 'metadata_token_end', 'metadata_char_start', 'metadata_char_end', 'metadata_total_tokens', 'metadata_unique_tokens', 'metadata_function_word_count', 'metadata_content_word_count', 'metadata_function_word_ratio']
```

## 8. Notes

- **No external API keys required** - uses public HuggingFace dataset `manu/project_gutenberg`
- **Hardware**: CPU only, ~500 MB RAM peak usage
- **Random seeds**: Not applicable (deterministic processing)
- **Source novel**: Project Gutenberg eBook #42671 (Pride and Prejudice by Jane Austen, 1813, Chapman edition)

## 9. Troubleshooting

If `data.py` fails to find the novel text, ensure `pride_and_prejudice_novel.txt` exists. If not, the script will attempt to extract from `pride_and_prejudice_raw.txt` which requires downloading from HuggingFace (internet connection needed).
