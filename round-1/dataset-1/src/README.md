# Rolling Zipfian Narrative Windows Dataset

This artifact creates a dataset of rolling windows from Jane Austen's *Pride and Prejudice* (Project Gutenberg eBook #42671), pre-processed for Zipf-Mandelbrot frequency distribution analysis of narrative temporal dynamics.

## Overview

- **Source**: Project Gutenberg via HuggingFace dataset `manu/project_gutenberg` (English split)
- **Novel**: *Pride and Prejudice* by Jane Austen (1813)
- **Method**: Sliding window tokenization with 750-token windows and 50-token step size
- **Total windows**: 2,453
- **Total tokens in novel**: ~123,367

## Files

| File | Description |
|------|-------------|
| `data.py` | Reproducible generation script |
| `full_data_out.json` | Complete dataset with 2,453 examples (13 MB) |
| `mini_data_out.json` | First 3 examples for quick testing (15 KB) |
| `preview_data_out.json` | First 3 examples with truncated strings for inspection (3 KB) |
| `pride_and_prejudice_novel.txt` | Clean novel text (675 KB, ~121K words) |
| `pride_and_prejudice_raw.txt` | Raw download from HF (includes license boilerplate) |
| `pyproject.toml` | Pinned dependencies for reproducibility |
| `reproducibility.md` | Step-by-step reproduction instructions |
| `.aii/manifest.yaml` | Artifact manifest with keep/delete decisions |
| `logs/` | Execution logs |

## Data Structure

Each example in `full_data_out.json` follows the `exp_sel_data_out` schema:

```json
{
  "dataset": "pride_and_prejudice_rolling_windows",
  "examples": [
    {
      "input": "{\"text\": \"CHAPTER I....\", \"token_start\": 0, \"token_end\": 750, \"char_start\": 0, \"char_end\": 3872}",
      "output": "{\"total_tokens\": 750, \"unique_tokens\": 287, \"function_word_count\": 432, \"content_word_count\": 318, \"function_word_ratio\": 0.576, \"type_token_ratio\": 0.383}",
      "metadata_window_index": 0,
      "metadata_token_start": 0,
      "metadata_token_end": 750,
      "metadata_char_start": 0,
      "metadata_char_end": 3872,
      "metadata_total_tokens": 750,
      "metadata_unique_tokens": 287,
      "metadata_function_word_count": 432,
      "metadata_content_word_count": 318,
      "metadata_function_word_ratio": 0.576
    }
  ]
}
```

## Linguistic Metadata

Each window provides:
- **total_tokens**: 750 (fixed window size)
- **unique_tokens**: Vocabulary size within window (type count)
- **function_word_count**: Count of NLTK English stopwords
- **content_word_count**: Non-stopword tokens
- **function_word_ratio**: Proportion of function words
- **type_token_ratio**: Unique / total tokens (lexical diversity)

## Restoring Removed Files

```bash
# Restore virtual environment
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python datasets nltk python-dotenv httpx jsonschema loguru
```

## Usage

```python
import json

with open('full_data_out.json', 'r') as f:
    data = json.load(f)

examples = data['datasets'][0]['examples']
for ex in examples:
    input_data = json.loads(ex['input'])
    output_data = json.loads(ex['output'])
    # Process window text and linguistic features
```

## Domain Alignment

This dataset supports the computational linguistics practice of analyzing narrative temporal dynamics through rolling Zipfian windows, controlling for syntactic frequency (function words) versus semantic topicality (content words) as recommended in the field handbook.
