#!/usr/bin/env python3
"""
Generate rolling Zipfian narrative windows dataset from Pride and Prejudice.

Uses already downloaded novel text, tokenizes into sliding windows,
and computes linguistic metadata.
"""

import json
import re
from pathlib import Path

from loguru import logger


def setup_logging():
    logger.remove()
    logger.add(
        "logs/dataset_generation.log",
        rotation="30 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )


def load_novel_text():
    """Load already extracted novel text."""
    novel_path = Path("pride_and_prejudice_novel.txt")
    if novel_path.exists():
        logger.info("Loading existing novel text...")
        with open(novel_path, "r") as f:
            text = f.read()
        logger.info(f"Loaded novel: {len(text)} chars, ~{len(text.split())} words")
        return text
    
    # Fallback: load raw and extract
    raw_path = Path("pride_and_prejudice_raw.txt")
    if raw_path.exists():
        logger.info("Loading raw text and extracting novel...")
        with open(raw_path, "r") as f:
            raw_text = f.read()
        return extract_novel_text(raw_text)
    
    raise FileNotFoundError("Neither novel nor raw text found. Run initial download first.")


def extract_novel_text(raw_text: str) -> str:
    """Extract the novel content between Project Gutenberg markers."""
    # Find start marker
    start_markers = [
        "***START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***",
        "*** START OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***",
    ]
    start_idx = -1
    for marker in start_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            start_idx = idx + len(marker)
            break
    
    if start_idx == -1:
        raise ValueError("Start marker not found")
    
    # Find end marker
    end_markers = [
        "***END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***",
        "*** END OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***",
    ]
    end_idx = -1
    for marker in end_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            end_idx = idx
            break
    
    if end_idx == -1:
        raise ValueError("End marker not found")
    
    # Find the actual novel start (first CHAPTER I.)
    novel_start = raw_text.find("CHAPTER I.\n", start_idx)
    if novel_start == -1:
        novel_start = start_idx
    
    novel_text = raw_text[novel_start:end_idx]
    logger.info(f"Extracted novel: {len(novel_text)} chars, ~{len(novel_text.split())} words")
    return novel_text


def tokenize_with_positions(text: str):
    """Tokenize text and return tokens with character positions."""
    tokens = []
    positions = []
    
    for match in re.finditer(r"\b\w+\b", text):
        token = match.group().lower()
        tokens.append(token)
        positions.append((match.start(), match.end()))
    
    return tokens, positions


def create_windows(tokens, positions, text, window_size=750, step_size=50, stop_words=None):
    """Create sliding windows with linguistic metadata."""
    if stop_words is None:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    
    windows = []
    max_start = len(tokens) - window_size + 1
    
    for i in range(0, max_start, step_size):
        window_tokens = tokens[i:i + window_size]
        window_positions = positions[i:i + window_size]
        
        start_char = window_positions[0][0]
        end_char = window_positions[-1][1]
        raw_text = text[start_char:end_char]
        
        function_count = sum(1 for t in window_tokens if t in stop_words)
        content_count = len(window_tokens) - function_count
        unique_tokens = len(set(window_tokens))
        
        windows.append({
            "window_index": len(windows),
            "token_start": i,
            "token_end": i + window_size,
            "char_start": start_char,
            "char_end": end_char,
            "raw_text": raw_text,
            "total_tokens": window_size,
            "unique_tokens": unique_tokens,
            "function_word_count": function_count,
            "content_word_count": content_count,
            "function_word_ratio": function_count / window_size,
        })
    
    logger.info(f"Created {len(windows)} windows")
    return windows


def build_dataset(windows):
    """Build dataset in exp_sel_data_out format."""
    dataset = {
        "dataset": "pride_and_prejudice_rolling_windows",
        "examples": []
    }
    
    for w in windows:
        input_json = json.dumps({
            "text": w["raw_text"],
            "token_start": w["token_start"],
            "token_end": w["token_end"],
            "char_start": w["char_start"],
            "char_end": w["char_end"],
        })
        
        output_json = json.dumps({
            "total_tokens": w["total_tokens"],
            "unique_tokens": w["unique_tokens"],
            "function_word_count": w["function_word_count"],
            "content_word_count": w["content_word_count"],
            "function_word_ratio": w["function_word_ratio"],
            "type_token_ratio": w["unique_tokens"] / w["total_tokens"],
        })
        
        example = {
            "input": input_json,
            "output": output_json,
            "metadata_window_index": w["window_index"],
            "metadata_token_start": w["token_start"],
            "metadata_token_end": w["token_end"],
            "metadata_char_start": w["char_start"],
            "metadata_char_end": w["char_end"],
            "metadata_total_tokens": w["total_tokens"],
            "metadata_unique_tokens": w["unique_tokens"],
            "metadata_function_word_count": w["function_word_count"],
            "metadata_content_word_count": w["content_word_count"],
            "metadata_function_word_ratio": w["function_word_ratio"],
        }
        dataset["examples"].append(example)
    
    return dataset


def save_outputs(dataset, output_dir="."):
    """Save full, mini, and preview versions."""
    output = {"datasets": [dataset]}
    
    # Full
    full_path = Path(output_dir) / "full_data_out.json"
    with open(full_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved full dataset: {full_path} ({full_path.stat().st_size / 1e6:.2f} MB)")
    
    # Mini (first 3)
    mini_output = {"datasets": [{"dataset": dataset["dataset"], "examples": dataset["examples"][:3]}]}
    mini_path = Path(output_dir) / "mini_data_out.json"
    with open(mini_path, "w") as f:
        json.dump(mini_output, f, indent=2)
    logger.info(f"Saved mini dataset: {mini_path}")
    
    # Preview (first 3, truncated strings)
    def truncate_value(value, max_len=200):
        if isinstance(value, list):
            return [truncate_value(item, max_len) for item in value[:3]]
        if isinstance(value, dict):
            return {k: truncate_value(v, max_len) for k, v in value.items()}
        if isinstance(value, str):
            if len(value) > max_len:
                return value[:max_len] + "..."
        return value
    
    preview_output = truncate_value(output)
    preview_path = Path(output_dir) / "preview_data_out.json"
    with open(preview_path, "w") as f:
        json.dump(preview_output, f, indent=2)
    logger.info(f"Saved preview dataset: {preview_path}")


@logger.catch(reraise=True)
def main():
    setup_logging()
    logger.info("Starting Pride and Prejudice rolling windows dataset generation")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Load novel text (already downloaded)
    novel_text = load_novel_text()
    
    # Tokenize
    tokens, positions = tokenize_with_positions(novel_text)
    logger.info(f"Tokenized: {len(tokens)} tokens")
    
    # Create windows
    windows = create_windows(tokens, positions, novel_text, window_size=750, step_size=50)
    
    # Build dataset
    dataset = build_dataset(windows)
    
    # Save outputs
    save_outputs(dataset)
    
    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    main()
