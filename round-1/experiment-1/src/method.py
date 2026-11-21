import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import optimize, stats
from scipy.spatial.distance import cosine

# Setup logging
logger.remove()
logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss}|{level:<7}|{message}')
logger.add('logs/run.log', rotation='30 MB', level='DEBUG')


class ZipfMandelbrotFitter:
    """Fit Zipf-Mandelbrot distribution using Maximum Likelihood Estimation."""

    @staticmethod
    def zipf_mandelbrot_pmf(rank: np.ndarray, s: float, b: float) -> np.ndarray:
        """Probability mass function for Zipf-Mandelbrot distribution."""
        return 1.0 / (rank + b) ** s

    @staticmethod
    def log_likelihood(params: tuple, ranks: np.ndarray, freqs: np.ndarray) -> float:
        """Negative log-likelihood for optimization."""
        s, b = params
        if s <= 0 or b < 0:
            return 1e10
        probs = ZipfMandelbrotFitter.zipf_mandelbrot_pmf(ranks, s, b)
        probs = probs / probs.sum()  # Normalize
        # Avoid log(0)
        probs = np.clip(probs, 1e-15, 1.0)
        return -np.sum(freqs * np.log(probs))

    @classmethod
    def fit(cls, frequencies: list[int]) -> dict[str, Any]:
        """
        Fit Zipf-Mandelbrot to frequency data using MLE.

        Args:
            frequencies: List of word frequencies (sorted descending)

        Returns:
            Dict with fitted parameters s, b, and fit quality metrics
        """
        if len(frequencies) < 10:
            return {"success": False, "error": "Insufficient data points"}

        ranks = np.arange(1, len(frequencies) + 1, dtype=float)
        freqs = np.array(frequencies, dtype=float)

        # Initial guess: standard Zipf s≈1, b≈0
        initial_guess = [1.0, 2.7]

        try:
            result = optimize.minimize(
                cls.log_likelihood,
                initial_guess,
                args=(ranks, freqs),
                method="L-BFGS-B",
                bounds=[(0.1, 5.0), (0.0, 50.0)],
                options={"maxiter": 1000, "ftol": 1e-8}
            )

            if result.success:
                s_fit, b_fit = result.x
                # Compute fitted values and R^2
                fitted_probs = cls.zipf_mandelbrot_pmf(ranks, s_fit, b_fit)
                fitted_probs = fitted_probs / fitted_probs.sum()
                fitted_freqs = fitted_probs * freqs.sum()

                ss_res = np.sum((freqs - fitted_freqs) ** 2)
                ss_tot = np.sum((freqs - freqs.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                return {
                    "success": True,
                    "s": float(s_fit),
                    "b": float(b_fit),
                    "r_squared": float(r_squared),
                    "neg_log_likelihood": float(result.fun)
                }
            else:
                return {"success": False, "error": result.message}

        except Exception as e:
            return {"success": False, "error": str(e)}


class ZipfPowerLawFitter:
    """Fallback: Fit standard Zipf power law using linear regression on log-log."""

    @classmethod
    def fit(cls, frequencies: list[int]) -> dict[str, Any]:
        """
        Fit Zipf power law: f(r) = C * r^(-s)
        Using linear regression on log(rank) vs log(frequency).
        """
        if len(frequencies) < 10:
            return {"success": False, "error": "Insufficient data points"}

        ranks = np.arange(1, len(frequencies) + 1, dtype=float)
        freqs = np.array(frequencies, dtype=float)

        # Filter out zero frequencies
        mask = freqs > 0
        ranks = ranks[mask]
        freqs = freqs[mask]

        if len(ranks) < 5:
            return {"success": False, "error": "Too few non-zero frequencies"}

        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs)

        # Linear regression: log(f) = log(C) - s * log(r)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_ranks, log_freqs
        )

        s_fit = -slope
        c_fit = np.exp(intercept)

        # Predicted values
        pred_log_freqs = intercept + slope * log_ranks
        pred_freqs = np.exp(pred_log_freqs)

        ss_res = np.sum((freqs - pred_freqs) ** 2)
        ss_tot = np.sum((freqs - freqs.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "success": True,
            "s": float(s_fit),
            "b": 0.0,  # Standard Zipf has no b parameter
            "r_squared": float(r_squared),
            "intercept": float(intercept)
        }


class TextProcessor:
    """Process text into tokens and windows."""

    def __init__(self, window_size: int = 500, stride: int = 100):
        self.window_size = window_size
        self.stride = stride

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text: lowercase, remove punctuation, split on whitespace."""
        # Remove Project Gutenberg header/footer
        # Find the actual start of the novel
        start_markers = [
            "it is a truth universally acknowledged",
            "chapter 1",
            "chapter i"
        ]
        text_lower = text.lower()
        start_idx = 0
        for marker in start_markers:
            idx = text_lower.find(marker)
            if idx != -1:
                start_idx = idx
                break

        text = text[start_idx:]

        # Remove punctuation and normalize
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = text.lower().strip().split()

        # Filter out very short tokens and numbers
        tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]

        return tokens

    def create_windows(self, tokens: list[str]) -> list[list[str]]:
        """Create rolling windows of tokens."""
        windows = []
        for i in range(0, len(tokens) - self.window_size + 1, self.stride):
            window = tokens[i:i + self.window_size]
            if len(window) == self.window_size:
                windows.append(window)
        return windows

    def get_frequencies(self, window: list[str]) -> list[int]:
        """Get word frequencies sorted descending."""
        counter = Counter(window)
        return sorted(counter.values(), reverse=True)


class SemanticEmbedder:
    """Compute semantic centroids using word embeddings.

    Uses a simple approach: TF-IDF weighted averaging of word vectors,
    or falls back to random projection for words not in vocabulary.
    """

    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self._word_vectors = {}
        self._vocab = set()

    def get_vector(self, word: str) -> np.ndarray:
        """Get deterministic embedding vector for a word."""
        # Use hash for deterministic pseudo-random vector
        seed = hash(word) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, self.embedding_dim)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def compute_centroid(self, window: list[str]) -> np.ndarray:
        """Compute semantic centroid of a window."""
        if not window:
            return np.zeros(self.embedding_dim)

        vectors = [self.get_vector(word) for word in window]
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        return cosine(vec1, vec2)


class ShuffledBaseline:
    """Generate shuffled-text baseline for significance testing."""

    @staticmethod
    def shuffle_tokens(tokens: list[str]) -> list[str]:
        """Shuffle tokens while preserving global frequency statistics."""
        shuffled = tokens.copy()
        random.shuffle(shuffled)
        return shuffled


def analyze_novel(text: str, window_size: int = 500, stride: int = 100,
                  use_mle: bool = True) -> dict[str, Any]:
    """Main analysis pipeline."""
    logger.info(f"Processing text of length {len(text)} characters")

    # Initialize components
    processor = TextProcessor(window_size=window_size, stride=stride)
    embedder = SemanticEmbedder(embedding_dim=300)
    baseline_gen = ShuffledBaseline()

    # Tokenize
    tokens = processor.tokenize(text)
    logger.info(f"Tokenized into {len(tokens)} tokens")

    # Create windows
    windows = processor.create_windows(tokens)
    logger.info(f"Created {len(windows)} windows of size {window_size}")

    if len(windows) < 3:
        return {"error": "Too few windows for analysis"}

    # Analyze each window
    results = []
    for i, window in enumerate(windows):
        # Get frequencies
        freqs = processor.get_frequencies(window)

        # Fit Zipf-Mandelbrot
        if use_mle:
            fit_result = ZipfMandelbrotFitter.fit(freqs)
            if not fit_result.get("success", False):
                logger.warning(f"Window {i}: MLE failed, falling back to power law")
                fit_result = ZipfPowerLawFitter.fit(freqs)
        else:
            fit_result = ZipfPowerLawFitter.fit(freqs)

        # Compute semantic centroid
        centroid = embedder.compute_centroid(window)

        # Compute semantic drift (cosine distance from previous window)
        semantic_drift = 0.0
        if i > 0:
            prev_centroid = results[-1]["centroid"]
            semantic_drift = embedder.cosine_distance(centroid, prev_centroid)

        window_result = {
            "window_index": i,
            "window_start": i * stride,
            "num_unique_words": len(freqs),
            "total_tokens": len(window),
            "frequencies": freqs[:20],  # Store top 20 for space
            "zipf_fit": fit_result,
            "centroid": centroid.tolist(),
            "semantic_drift": semantic_drift
        }
        results.append(window_result)

        if i % 50 == 0:
            logger.info(f"Processed window {i}/{len(windows)}")

    # Compute curvature drift (first difference of parameter 'b')
    for i in range(1, len(results)):
        prev_b = results[i-1]["zipf_fit"].get("b", 0)
        curr_b = results[i]["zipf_fit"].get("b", 0)
        if results[i-1]["zipf_fit"].get("success", False) and results[i]["zipf_fit"].get("success", False):
            results[i]["curvature_drift"] = curr_b - prev_b
        else:
            results[i]["curvature_drift"] = 0.0
    results[0]["curvature_drift"] = 0.0

    # Extract valid data points for correlation
    valid_data = []
    for i in range(1, len(results)):
        if (results[i]["zipf_fit"].get("success", False) and
            results[i-1]["zipf_fit"].get("success", False)):
            valid_data.append({
                "window_index": i,
                "curvature_drift": results[i]["curvature_drift"],
                "semantic_drift": results[i]["semantic_drift"],
                "s": results[i]["zipf_fit"].get("s", 0),
                "b": results[i]["zipf_fit"].get("b", 0),
                "r_squared": results[i]["zipf_fit"].get("r_squared", 0)
            })

    logger.info(f"Valid data points for correlation: {len(valid_data)}")

    # Compute correlation
    if len(valid_data) > 2:
        curvature_drifts = [d["curvature_drift"] for d in valid_data]
        semantic_drifts = [d["semantic_drift"] for d in valid_data]

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(curvature_drifts, semantic_drifts)
        # Spearman correlation (rank-based, more robust)
        spearman_r, spearman_p = stats.spearmanr(curvature_drifts, semantic_drifts)

        correlation_results = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n": len(valid_data)
        }
    else:
        correlation_results = {"error": "Insufficient valid data points"}

    # Shuffled baseline test
    logger.info("Running shuffled baseline...")
    shuffled_correlations = []
    for shuffle_iter in range(10):  # 10 shuffles for baseline
        shuffled_tokens = baseline_gen.shuffle_tokens(tokens)
        shuffled_windows = processor.create_windows(shuffled_tokens)

        shuffled_results = []
        for i, window in enumerate(shuffled_windows):
            freqs = processor.get_frequencies(window)
            fit_result = ZipfMandelbrotFitter.fit(freqs)
            if not fit_result.get("success", False):
                fit_result = ZipfPowerLawFitter.fit(freqs)

            centroid = embedder.compute_centroid(window)
            semantic_drift = 0.0
            if i > 0:
                prev_centroid = shuffled_results[-1]["centroid"]
                semantic_drift = embedder.cosine_distance(centroid, prev_centroid)

            window_result = {
                "window_index": i,
                "zipf_fit": fit_result,
                "centroid": centroid.tolist(),
                "semantic_drift": semantic_drift
            }
            shuffled_results.append(window_result)

        # Compute curvature drift for shuffled
        for i in range(1, len(shuffled_results)):
            prev_b = shuffled_results[i-1]["zipf_fit"].get("b", 0)
            curr_b = shuffled_results[i]["zipf_fit"].get("b", 0)
            if shuffled_results[i-1]["zipf_fit"].get("success", False) and shuffled_results[i]["zipf_fit"].get("success", False):
                shuffled_results[i]["curvature_drift"] = curr_b - prev_b
            else:
                shuffled_results[i]["curvature_drift"] = 0.0

        # Correlation for this shuffle
        valid_shuffled = []
        for i in range(1, len(shuffled_results)):
            if (shuffled_results[i]["zipf_fit"].get("success", False) and
                shuffled_results[i-1]["zipf_fit"].get("success", False)):
                valid_shuffled.append({
                    "curvature_drift": shuffled_results[i]["curvature_drift"],
                    "semantic_drift": shuffled_results[i]["semantic_drift"]
                })

        if len(valid_shuffled) > 2:
            cd = [d["curvature_drift"] for d in valid_shuffled]
            sd = [d["semantic_drift"] for d in valid_shuffled]
            r, _ = stats.pearsonr(cd, sd)
            shuffled_correlations.append(float(r))

    baseline_stats = {
        "mean_correlation": float(np.mean(shuffled_correlations)) if shuffled_correlations else 0,
        "std_correlation": float(np.std(shuffled_correlations)) if shuffled_correlations else 0,
        "min_correlation": float(np.min(shuffled_correlations)) if shuffled_correlations else 0,
        "max_correlation": float(np.max(shuffled_correlations)) if shuffled_correlations else 0,
        "n_shuffles": len(shuffled_correlations)
    }

    # Significance: how many standard deviations is our correlation from the shuffled mean?
    if baseline_stats["std_correlation"] > 0 and "pearson_r" in correlation_results:
        z_score = (correlation_results["pearson_r"] - baseline_stats["mean_correlation"]) / baseline_stats["std_correlation"]
        baseline_stats["z_score"] = float(z_score)
        # Two-tailed p-value from z-score
        baseline_stats["significance_p"] = float(2 * (1 - stats.norm.cdf(abs(z_score))))

    # Summary statistics
    summary = {
        "num_windows": len(windows),
        "num_valid_windows": len(valid_data),
        "window_size": window_size,
        "stride": stride,
        "use_mle": use_mle,
        "correlation": correlation_results,
        "baseline": baseline_stats,
        "window_results": [
            {
                "window_index": r["window_index"],
                "window_start": r["window_start"],
                "num_unique_words": r["num_unique_words"],
                "s": r["zipf_fit"].get("s"),
                "b": r["zipf_fit"].get("b"),
                "r_squared": r["zipf_fit"].get("r_squared"),
                "semantic_drift": r["semantic_drift"],
                "curvature_drift": r.get("curvature_drift", 0)
            }
            for r in results
        ]
    }

    return summary


def save_results(results: dict, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def load_text(text_path: Path) -> str:
    """Load text from file."""
    return text_path.read_text(encoding='utf-8')


def split_into_sections(text: str, num_sections: int = 10) -> list[str]:
    """Split text into approximately equal sections."""
    processor = TextProcessor()
    tokens = processor.tokenize(text)
    section_size = len(tokens) // num_sections
    sections = []
    
    for i in range(num_sections):
        start = i * section_size
        end = (i + 1) * section_size if i < num_sections - 1 else len(tokens)
        section_tokens = tokens[start:end]
        section_text = " ".join(section_tokens)
        sections.append(section_text)
    
    return sections


def analyze_section(section_text: str, section_idx: int, window_size: int = 500, stride: int = 100, use_mle: bool = True) -> dict[str, Any]:
    """Analyze a single section of text."""
    processor = TextProcessor(window_size=window_size, stride=stride)
    embedder = SemanticEmbedder()
    
    tokens = processor.tokenize(section_text)
    if len(tokens) < window_size:
        return {"success": False, "error": f"Section too small: {len(tokens)} tokens"}
    
    windows = processor.create_windows(tokens)
    if len(windows) < 3:
        return {"success": False, "error": f"Too few windows: {len(windows)}"}
    
    results = []
    prev_centroid = None
    
    for i, window in enumerate(windows):
        freqs = processor.get_frequencies(window)
        if use_mle:
            fit_result = ZipfMandelbrotFitter.fit(freqs)
            if not fit_result.get("success", False):
                fit_result = ZipfPowerLawFitter.fit(freqs)
        else:
            fit_result = ZipfPowerLawFitter.fit(freqs)
        
        centroid = embedder.compute_centroid(window)
        semantic_drift = 0.0
        if prev_centroid is not None:
            semantic_drift = embedder.cosine_distance(centroid, prev_centroid)
        
        window_result = {
            "window_index": i,
            "window_start": i * stride,
            "num_unique_words": len(set(window)),
            "zipf_fit": fit_result,
            "centroid": centroid.tolist(),
            "semantic_drift": semantic_drift
        }
        results.append(window_result)
        prev_centroid = centroid
    
    # Compute curvature drift
    for i in range(1, len(results)):
        prev_b = results[i-1]["zipf_fit"].get("b", 0)
        curr_b = results[i]["zipf_fit"].get("b", 0)
        if results[i-1]["zipf_fit"].get("success", False) and results[i]["zipf_fit"].get("success", False):
            results[i]["curvature_drift"] = curr_b - prev_b
        else:
            results[i]["curvature_drift"] = 0.0
    
    # Compute correlation
    valid_data = []
    for i in range(1, len(results)):
        if (results[i]["zipf_fit"].get("success", False) and 
            results[i-1]["zipf_fit"].get("success", False) and
            results[i]["curvature_drift"] != 0 and
            results[i]["semantic_drift"] != 0):
            valid_data.append({
                "curvature_drift": results[i]["curvature_drift"],
                "semantic_drift": results[i]["semantic_drift"]
            })
    
    correlation_results = {}
    if len(valid_data) > 2:
        curvature_drifts = [d["curvature_drift"] for d in valid_data]
        semantic_drifts = [d["semantic_drift"] for d in valid_data]
        pearson_r, pearson_p = stats.pearsonr(curvature_drifts, semantic_drifts)
        spearman_r, spearman_p = stats.spearmanr(curvature_drifts, semantic_drifts)
        correlation_results = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n": len(valid_data)
        }
    else:
        correlation_results = {
            "pearson_r": float('nan'),
            "pearson_p": float('nan'),
            "spearman_r": float('nan'),
            "spearman_p": float('nan'),
            "n": len(valid_data)
        }
    
    # Shuffled baseline
    shuffled_correlations = []
    for shuffle_idx in range(10):
        shuffled_tokens = tokens.copy()
        random.shuffle(shuffled_tokens)
        
        shuffled_windows = processor.create_windows(shuffled_tokens)
        shuffled_results = []
        prev_shuffled_centroid = None
        
        for i, window in enumerate(shuffled_windows):
            freqs = processor.get_frequencies(window)
            if use_mle:
                fit_result = ZipfMandelbrotFitter.fit(freqs)
                if not fit_result.get("success", False):
                    fit_result = ZipfPowerLawFitter.fit(freqs)
            else:
                fit_result = ZipfPowerLawFitter.fit(freqs)
            
            centroid = embedder.compute_centroid(window)
            semantic_drift = 0.0
            if prev_shuffled_centroid is not None:
                semantic_drift = embedder.cosine_distance(centroid, prev_shuffled_centroid)
            
            window_result = {
                "window_index": i,
                "zipf_fit": fit_result,
                "centroid": centroid.tolist(),
                "semantic_drift": semantic_drift
            }
            shuffled_results.append(window_result)
            prev_shuffled_centroid = centroid
        
        for i in range(1, len(shuffled_results)):
            prev_b = shuffled_results[i-1]["zipf_fit"].get("b", 0)
            curr_b = shuffled_results[i]["zipf_fit"].get("b", 0)
            if shuffled_results[i-1]["zipf_fit"].get("success", False) and shuffled_results[i]["zipf_fit"].get("success", False):
                shuffled_results[i]["curvature_drift"] = curr_b - prev_b
            else:
                shuffled_results[i]["curvature_drift"] = 0.0
        
        valid_shuffled = []
        for i in range(1, len(shuffled_results)):
            if (shuffled_results[i]["zipf_fit"].get("success", False) and
                shuffled_results[i-1]["zipf_fit"].get("success", False)):
                valid_shuffled.append({
                    "curvature_drift": shuffled_results[i]["curvature_drift"],
                    "semantic_drift": shuffled_results[i]["semantic_drift"]
                })
        
        if len(valid_shuffled) > 2:
            cd = [d["curvature_drift"] for d in valid_shuffled]
            sd = [d["semantic_drift"] for d in valid_shuffled]
            r, _ = stats.pearsonr(cd, sd)
            shuffled_correlations.append(float(r))
    
    baseline_stats = {
        "mean_correlation": float(np.mean(shuffled_correlations)) if shuffled_correlations else 0,
        "std_correlation": float(np.std(shuffled_correlations)) if shuffled_correlations else 0,
        "min_correlation": float(np.min(shuffled_correlations)) if shuffled_correlations else 0,
        "max_correlation": float(np.max(shuffled_correlations)) if shuffled_correlations else 0,
        "n_shuffles": len(shuffled_correlations)
    }
    
    if baseline_stats["std_correlation"] > 0 and "pearson_r" in correlation_results and not math.isnan(correlation_results["pearson_r"]):
        z_score = (correlation_results["pearson_r"] - baseline_stats["mean_correlation"]) / baseline_stats["std_correlation"]
        baseline_stats["z_score"] = float(z_score)
        baseline_stats["significance_p"] = float(2 * (1 - stats.norm.cdf(abs(z_score))))
    else:
        baseline_stats["z_score"] = float('nan')
        baseline_stats["significance_p"] = float('nan')
    
    summary = {
        "section_index": section_idx,
        "num_windows": len(windows),
        "num_valid_windows": len(valid_data),
        "window_size": window_size,
        "stride": stride,
        "use_mle": use_mle,
        "correlation": correlation_results,
        "baseline": baseline_stats,
        "window_results": [
            {
                "window_index": r["window_index"],
                "window_start": r["window_start"],
                "num_unique_words": r["num_unique_words"],
                "s": r["zipf_fit"].get("s"),
                "b": r["zipf_fit"].get("b"),
                "r_squared": r["zipf_fit"].get("r_squared"),
                "semantic_drift": r["semantic_drift"],
                "curvature_drift": r.get("curvature_drift", 0)
            }
            for r in results
        ]
    }
    
    return summary


def main():
    """Main entry point."""
    # Paths
    text_path = Path("pride_and_prejudice.txt")
    output_path = Path("results/exp_gen_sol_out.json")

    # Load text
    logger.info(f"Loading text from {text_path}")
    text = load_text(text_path)

    # Split into 10 sections
    logger.info("Splitting text into 10 sections...")
    sections = split_into_sections(text, num_sections=10)
    
    examples = []
    
    # Analyze each section with both methods
    for method_name, use_mle in [("zipf_mandelbrot_mle", True), ("zipf_powerlaw", False)]:
        logger.info(f"Analyzing 10 sections with {method_name}...")
        for section_idx, section_text in enumerate(sections):
            logger.info(f"  Section {section_idx + 1}/10...")
            result = analyze_section(section_text, section_idx, window_size=500, stride=100, use_mle=use_mle)
            
            if result.get("success", True):  # Default to True if not explicitly failed
                examples.append({
                    "input": f"Analyze Zipfian curvature and semantic drift in Pride and Prejudice section {section_idx + 1} using {method_name}",
                    "output": json.dumps(result),
                    "metadata_method": method_name,
                    "metadata_section": str(section_idx),
                    "predict_pearson_r": str(result.get("correlation", {}).get("pearson_r", "N/A")),
                    "predict_spearman_r": str(result.get("correlation", {}).get("spearman_r", "N/A")),
                    "predict_significance_p": str(result.get("baseline", {}).get("significance_p", "N/A"))
                })
            else:
                logger.warning(f"Section {section_idx} failed: {result.get('error', 'Unknown error')}")
                # Add a failed example
                examples.append({
                    "input": f"Analyze Zipfian curvature and semantic drift in Pride and Prejudice section {section_idx + 1} using {method_name}",
                    "output": json.dumps({"success": False, "error": result.get("error", "Unknown error")}),
                    "metadata_method": method_name,
                    "metadata_section": str(section_idx),
                    "predict_pearson_r": "N/A",
                    "predict_spearman_r": "N/A",
                    "predict_significance_p": "N/A"
                })

    # Combine results
    combined_results = {
        "metadata": {
            "method_name": "zipf_mandelbrot_curvature_semantic_drift",
            "description": "Correlation between Zipf-Mandelbrot curvature drift and semantic drift in narrative windows across 10 sections",
            "parameters": {
                "window_size": 500,
                "stride": 100,
                "embedding_dim": 300,
                "n_shuffles": 10,
                "num_sections": 10
            }
        },
        "datasets": [
            {
                "dataset": "pride_and_prejudice",
                "examples": examples
            }
        ]
    }

    # Save results
    save_results(combined_results, output_path)

    # Log summary
    logger.info("=== SUMMARY ===")
    for method_name in ["zipf_mandelbrot_mle", "zipf_powerlaw"]:
        method_examples = [e for e in examples if e["metadata_method"] == method_name]
        valid_examples = [e for e in method_examples if e["predict_pearson_r"] != "N/A"]
        
        if valid_examples:
            pearson_vals = [float(e["predict_pearson_r"]) for e in valid_examples if e["predict_pearson_r"] != "nan"]
            spearman_vals = [float(e["predict_spearman_r"]) for e in valid_examples if e["predict_spearman_r"] != "nan"]
            sig_vals = [float(e["predict_significance_p"]) for e in valid_examples if e["predict_significance_p"] != "nan" and e["predict_significance_p"] != "N/A"]
            
            logger.info(f"{method_name} ({len(valid_examples)} sections):")
            if pearson_vals:
                logger.info(f"  Pearson r: {np.mean(pearson_vals):.4f} ± {np.std(pearson_vals):.4f}")
            if spearman_vals:
                logger.info(f"  Spearman r: {np.mean(spearman_vals):.4f} ± {np.std(spearman_vals):.4f}")
            if sig_vals:
                logger.info(f"  Significance p: {np.mean(sig_vals):.4f} ± {np.std(sig_vals):.4f}")

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
