import os
import re
import csv
import unicodedata
import logging
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher, ndiff
from fuzzywuzzy import fuzz
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer

# =======================
# Configuration Section
# =======================
CONFIG = {  # DO NOT REMOVE COMMENTED METRICS
    "files": {
        "primary_input": "truncated_csv.csv",
        "bart_input": "processed_1000_testing_csv.csv",
        "output": "updated_file_comma.csv",
        "output_sample": "sample_updated_file_comma2.csv",
    },
    "separators": {
        "primary_input": ",",
        "bart_input": ",",
        "output": ",",
    },
    "thresholds": {
        "difflib": 0.93,
        "fuzzywuzzy": 93,
        "jaccard": 0.93,
    },
    "transcription_columns": [
        'pyte_ocr', 'chatgpt_ocr', 'LLAMA32_BASE',
        'BART_untuned', 'BART_gold_100', 'BART_gold_1000', 'BART_gold_10000',
        'BART_silver_100', 'BART_silver_1000', 'BART_silver_10000'
    ],
    "metrics": [
        # 'similarity_difflib',     # SequenceMatcher ratio
        # 'fuzz_ratio',             # FuzzyWuzzy ratio
        # 'jaccard_similarity',     # Jaccard index
        # 'bleu_score',             # BLEU score for translation
        # 'word_error_rate',        # WER for transcription accuracy
        'precision',                # Precision metric for retrieval
        'recall'                    # Recall metric for retrieval
    ],
    "text_processing": {
        "max_char_length": 5000,
        "max_diff_length": 1000,
        "phrases_to_delete": [  # Phrases to remove from texts
            "correct this ocr",
            "correct this ocr:"
        ],
    },
    "sample": {
        "size": 100,
    },
    "debugging": {
        "drop_after_1000": True,
    },
    "output_dirs": {
        "plots": "plots",
        "summary_statistics": "summary_statistics",
        "histogram_data": "histogram_data",
        "histogram_plots": "histogram_plots",
    },
}

# =======================
# Setup Logging
# =======================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize tqdm for pandas
tqdm.pandas(desc="Progress")

# =======================
# Helper Functions
# =======================

def read_csv(file_path: str, separator: str) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep=separator, dtype=str)
        logger.info(f"Successfully read '{file_path}' with separator '{separator}'.")
        return df
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        raise
    except Exception as e:
        logger.error(f"Error reading '{file_path}': {e}")
        raise

def display_df_info(df: pd.DataFrame, name: str) -> None:
    """
    Logs basic information about the DataFrame.
    """
    logger.info(f"--- {name} ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data Types:\n{df.dtypes}\n")

def normalize_text(text: str, max_chars: int, phrases_to_delete: List[str] = []) -> str:
    """
    Normalizes text by:
    - Deleting specified unwanted phrases
    - Decoding escape sequences (e.g., '\\n', '\\t', '\\r', '\\\\')
    - Lowercasing
    - Removing accents
    - Removing punctuation
    - Replacing newlines and other whitespace with spaces
    - Removing extra whitespace
    - Truncating or padding to a maximum length
    
    Args:
        text (str): The text to normalize.
        max_chars (int): Maximum character length.
        phrases_to_delete (List[str]): List of phrases to remove from the text.
    
    Returns:
        str: The normalized text.
    """
    if not isinstance(text, str):
        return ' ' * max_chars

    # Step 1: Delete unwanted phrases
    for phrase in phrases_to_delete:
        # Use regex to remove the phrase case-insensitively
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub('', text)
    
    # Step 2: Decode escape sequences (e.g., '\\n' -> '\n', '\\t' -> '\t', '\\\\' -> '\\')
    try:
        # Replace double backslashes with single backslashes to ensure correct decoding
        text = text.replace('\\\\', '\\')
        text = text.encode('utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        # If decoding fails, proceed without decoding to avoid data loss
        pass

    # Step 3: Normalize Unicode characters to NFC form and convert to lowercase
    text = unicodedata.normalize('NFC', text).lower()

    # Step 4: Remove accents and diacritics
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Step 5: Remove punctuation (retain word characters and spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Step 6: Replace all whitespace characters (including newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: Truncate or pad the text to the desired length
    if len(text) > max_chars:
        text = text[:max_chars - 15] + '... [truncated]'
    else:
        text = text.ljust(max_chars)

    return text

def compute_similarity_difflib(text1: str, text2: str) -> float:
    """
    Computes similarity ratio using difflib's SequenceMatcher.
    """
    return SequenceMatcher(None, text1, text2).ratio()

def compute_similarity_fuzzywuzzy(text1: str, text2: str) -> Dict[str, int]:
    """
    Computes similarity ratios using FuzzyWuzzy's different methods.
    """
    return {
        'fuzz_ratio': fuzz.ratio(text1, text2),
        'fuzz_partial_ratio': fuzz.partial_ratio(text1, text2),
        'fuzz_token_sort_ratio': fuzz.token_sort_ratio(text1, text2),
        'fuzz_token_set_ratio': fuzz.token_set_ratio(text1, text2)
    }

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Computes Jaccard similarity index between two texts.
    """
    set1, set2 = set(text1.split()), set(text2.split())
    union = set1.union(set2)
    intersection = set1.intersection(set2)
    return len(intersection) / len(union) if union else 0.0

def compute_bleu_score(text1: str, text2: str) -> float:
    """
    Computes BLEU score between two texts.
    """
    # Tokenize the texts
    reference = re.findall(r'\w+', text1.lower())
    hypothesis = re.findall(r'\w+', text2.lower())
    
    if not reference or not hypothesis:
        return 0.0
    
    # Use smoothing to handle cases with zero counts
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
    return score

def compute_word_error_rate(text1: str, text2: str) -> float:
    """
    Computes Word Error Rate (WER) between two texts.
    """
    return wer(text1, text2)

def compute_precision(text1: str, text2: str) -> float:
    """
    Computes Precision: the proportion of words in text2 that are also in text1.
    
    Args:
        text1 (str): The first text (e.g., ground truth).
        text2 (str): The second text (e.g., prediction).
    
    Returns:
        float: The precision score.
    """
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    return len(intersection) / len(set2) if set2 else 0.0

def compute_recall(text1: str, text2: str) -> float:
    """
    Computes Recall: the proportion of words in text1 that are also in text2.
    
    Args:
        text1 (str): The first text (e.g., ground truth).
        text2 (str): The second text (e.g., prediction).
    
    Returns:
        float: The recall score.
    """
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    return len(intersection) / len(set1) if set1 else 0.0

def get_differences(text1: str, text2: str, max_length: int) -> str:
    """
    Generates a string showing differences between two texts using difflib's ndiff.
    """
    diff = list(ndiff(text1.split(), text2.split()))
    changes = [d for d in diff if d.startswith('- ') or d.startswith('+ ')]
    differences = ' '.join(changes)
    return differences[:max_length] + '... [truncated]' if len(differences) > max_length else differences

def compare_texts(row: pd.Series, method_col: str, thresholds: Dict[str, float], max_diff_length: int) -> pd.Series:
    """
    Compares normalized transcription with a specific method's transcription.
    Computes similarities using difflib, FuzzyWuzzy, Jaccard similarity, BLEU, WER, Precision, and Recall.
    """
    human_text = row['normalized_transcription'].strip() if isinstance(row['normalized_transcription'], str) else ''
    method_text = row[f'normalized_{method_col}'].strip() if isinstance(row[f'normalized_{method_col}'], str) else ''
    
    # Compute similarities
    similarity_difflib = compute_similarity_difflib(human_text, method_text)
    equal_difflib = similarity_difflib >= thresholds['difflib']
    
    similarity_fuzzy = compute_similarity_fuzzywuzzy(human_text, method_text)
    equal_fuzzy = similarity_fuzzy['fuzz_ratio'] >= thresholds['fuzzywuzzy']
    
    similarity_jaccard = compute_jaccard_similarity(human_text, method_text)
    equal_jaccard = similarity_jaccard >= thresholds['jaccard']
    
    similarity_bleu = compute_bleu_score(human_text, method_text)
    similarity_wer = compute_word_error_rate(human_text, method_text)
    
    # Compute Precision and Recall
    similarity_precision = compute_precision(human_text, method_text)
    similarity_recall = compute_recall(human_text, method_text)
    
    differences = get_differences(human_text, method_text, max_diff_length)
    
    # Compile results
    results = {
        f'{method_col}_equal_difflib': equal_difflib,
        f'{method_col}_similarity_difflib': similarity_difflib,
        f'{method_col}_fuzz_ratio': similarity_fuzzy['fuzz_ratio'],
        f'{method_col}_fuzz_partial_ratio': similarity_fuzzy['fuzz_partial_ratio'],
        f'{method_col}_fuzz_token_sort_ratio': similarity_fuzzy['fuzz_token_sort_ratio'],
        f'{method_col}_fuzz_token_set_ratio': similarity_fuzzy['fuzz_token_set_ratio'],
        f'{method_col}_equal_fuzzywuzzy': equal_fuzzy,
        f'{method_col}_jaccard_similarity': similarity_jaccard,
        f'{method_col}_equal_jaccard': equal_jaccard,
        f'{method_col}_bleu_score': similarity_bleu,
        f'{method_col}_word_error_rate': similarity_wer,
        f'{method_col}_precision': similarity_precision,  # Added Precision
        f'{method_col}_recall': similarity_recall,        # Added Recall
        f'{method_col}_differences': differences
    }
    
    return pd.Series(results)

def detect_repetition(text: str, phrase_length: int = 3, repetition_threshold: int = 3) -> bool:
    """
    Detects if any phrase of a given length repeats more than the repetition threshold.
    
    Args:
        text (str): The text to analyze.
        phrase_length (int): Number of words in each phrase.
        repetition_threshold (int): Number of allowed repetitions before flagging.
        
    Returns:
        bool: True if repetition exceeds threshold, else False.
    """
    phrases = re.findall(r'\b(?:\w+\s+){%d}\w+\b' % (phrase_length - 1), text.lower())
    phrase_counts = {}
    for phrase in phrases:
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        if phrase_counts[phrase] > repetition_threshold:
            return True
    return False

def check_character_uniqueness(text: str, uniqueness_threshold: float = 0.4) -> bool:
    """
    Checks if the ratio of unique characters to total characters is below the threshold.
    
    Args:
        text (str): The text to analyze.
        uniqueness_threshold (float): Threshold below which uniqueness is considered low.
        
    Returns:
        bool: True if uniqueness is below threshold, else False.
    """
    if not text:
        return False
    unique_chars = set(text)
    uniqueness_ratio = len(unique_chars) / len(text)
    return uniqueness_ratio < uniqueness_threshold

def flag_repeated_text(row: pd.Series) -> int:
    """
    Flags the row as 'repeated_text' if both repetition and uniqueness criteria are met.
    
    Args:
        row (pd.Series): A row of the DataFrame.
        
    Returns:
        int: 1 if flagged as repeated_text, else 0.
    """
    text = row['normalized_transcription']
    is_repetition = detect_repetition(text)
    is_low_uniqueness = check_character_uniqueness(text)
    return 1 if (is_repetition and is_low_uniqueness) else 0

def ensure_length(df: pd.DataFrame, methods: List[str], max_chars: int) -> pd.DataFrame:
    """
    Ensures all normalized transcription columns are limited to max_chars.
    """
    for method in methods:
        norm_col = f'normalized_{method}'
        if norm_col in df.columns:
            df[norm_col] = df[norm_col].apply(
                lambda x: x if isinstance(x, str) and len(x) <= max_chars else (x[:max_chars - 15] + '... [truncated]' if isinstance(x, str) else x)
            )
    return df

def plot_overlayed_histograms(df: pd.DataFrame, methods: List[str], metric: str, output_dir: str) -> None:
    """
    Plots overlayed histograms for a given metric across different OCR methods.
    Annotates medians next to the model names.
    """
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("hsv", len(methods))
    
    for method, color in zip(methods, palette):
        column_name = f'{method}_{metric}'
        if column_name in df.columns:
            median_val = df[column_name].median()
            sns.histplot(
                df[column_name],
                bins=30,
                kde=True,
                label=f"{method} (Median: {median_val:.2f})",
                color=color,
                stat="density",
                common_norm=False,
                alpha=0.6
            )
    
    plt.title(f'Overlayed Histogram of {metric.replace("_", " ").title()}')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Density')
    plt.legend(title="Models (Median Values)")
    plt.tight_layout()
    filename = f'overlayed_histogram_{metric}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    logger.info(f"Saved overlayed histogram for '{metric}' as '{filename}'.")

def plot_top_bottom_models(df: pd.DataFrame, methods: List[str], metric: str, output_dir: str) -> None:
    """
    Plots histograms for the top 3 and bottom 3 models based on mean similarity for a given metric.
    """
    # Calculate mean similarity for each method
    mean_scores = {}
    for method in methods:
        column = f'{method}_{metric}'
        if column in df.columns:
            mean_scores[method] = df[column].mean()
    
    # Sort methods based on mean similarity
    sorted_methods = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    
    top_3 = [method for method, score in sorted_methods[:3]]
    bottom_3 = [method for method, score in sorted_methods[-3:]]
    
    # Plot Top 3
    plt.figure(figsize=(12, 8))
    palette_top = sns.color_palette("viridis", len(top_3))
    for method, color in zip(top_3, palette_top):
        column_name = f'{method}_{metric}'
        if column_name in df.columns:
            sns.histplot(
                df[column_name],
                bins=30,
                kde=True,
                label=f"{method} (Mean: {mean_scores[method]:.2f})",
                color=color,
                stat="density",
                common_norm=False,
                alpha=0.6
            )
    plt.title(f'Top 3 Models for {metric.replace("_", " ").title()}')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Density')
    plt.legend(title="Top 3 Models (Mean Values)")
    plt.tight_layout()
    filename_top = f'top3_histogram_{metric}.png'
    plt.savefig(os.path.join(output_dir, filename_top))
    plt.close()
    logger.info(f"Saved top 3 histogram for '{metric}' as '{filename_top}'.")
    
    # Plot Bottom 3
    plt.figure(figsize=(12, 8))
    palette_bottom = sns.color_palette("magma", len(bottom_3))
    for method, color in zip(bottom_3, palette_bottom):
        column_name = f'{method}_{metric}'
        if column_name in df.columns:
            sns.histplot(
                df[column_name],
                bins=30,
                kde=True,
                label=f"{method} (Mean: {mean_scores[method]:.2f})",
                color=color,
                stat="density",
                common_norm=False,
                alpha=0.6
            )
    plt.title(f'Bottom 3 Models for {metric.replace("_", " ").title()}')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Density')
    plt.legend(title="Bottom 3 Models (Mean Values)")
    plt.tight_layout()
    filename_bottom = f'bottom3_histogram_{metric}.png'
    plt.savefig(os.path.join(output_dir, filename_bottom))
    plt.close()
    logger.info(f"Saved bottom 3 histogram for '{metric}' as '{filename_bottom}'.")

def generate_summary_statistics(df: pd.DataFrame, methods: List[str], metrics: List[str], output_dir: str) -> None:
    """
    Generates and saves summary statistics for each measurement and model.
    Formats the statistics as percentages with two decimal places.
    """
    for metric in metrics:
        summary_data = {}
        for method in methods:
            column = f'{method}_{metric}'
            if column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    summary_data[method] = {
                        'Mean': f"{df[column].mean() * 100:.0f}",
                        'Median': f"{df[column].median() * 100:.0f}",
                        'Std Dev': f"{df[column].std() * 100:.0f}",
                        'Min': f"{df[column].min() * 100:.0f}",
                        'Max': f"{df[column].max() * 100:.0f}",
                        '25%': f"{df[column].quantile(0.25) * 100:.0f}",
                        '75%': f"{df[column].quantile(0.75) * 100:.0f}"
                    }
                else:
                    logger.warning(f"Column '{column}' is not numeric and will be skipped in summary statistics.")
        if summary_data:
            summary_df = pd.DataFrame(summary_data).T[['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25%', '75%']]
            summary_csv = os.path.join(output_dir, f'summary_statistics_{metric}.csv')
            summary_df.to_csv(summary_csv)
            logger.info(f"Saved summary statistics for '{metric}' as '{summary_csv}'.")
            logger.info(f"\nSummary Statistics for {metric}:\n{summary_df}\n")

def save_csv(df: pd.DataFrame, file_path: str, separator: str) -> None:
    """
    Saves the DataFrame to a CSV file with specified separator and quoting.
    """
    try:
        df.to_csv(
            file_path,
            index=False,
            sep=separator,
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            quotechar='"',
            doublequote=True,
            encoding='utf-8'
        )
        logger.info(f"Saved DataFrame to '{file_path}'.")
    except Exception as e:
        logger.error(f"Error saving DataFrame to '{file_path}': {e}")
        raise

def create_output_dirs(directories: List[str]) -> None:
    """
    Creates output directories if they do not exist.
    """
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured existence of directory '{dir_path}'.")

def process_sample(df: pd.DataFrame, sample_size: int, output_path: str, separator: str) -> None:
    """
    Creates and saves a sample CSV with the first 'sample_size' rows.
    """
    try:
        df_sample = df.head(sample_size)
        save_csv(df_sample, output_path, separator)
        logger.info(f"Sample CSV with first {sample_size} rows saved as '{output_path}'.")
    except Exception as e:
        logger.error(f"Error creating sample CSV: {e}")
        raise

def plot_bar_chart(df: pd.DataFrame, methods: List[str], metric: str, output_dir: str) -> None:
    """
    Plots a bar chart for a specific metric across different OCR models.
    Each bar represents a model's performance on the metric.
    The y-axis ranges from 0 to 100%, and each bar has its value displayed above it.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the metrics.
        methods (List[str]): List of OCR model names in the desired order.
        metric (str): The metric to plot (e.g., 'precision', 'recall').
        output_dir (str): Directory where the plot will be saved.
    """
    # Extract metric values for each method and convert to percentages
    data = []
    for method in methods:
        column = f'{method}_{metric}'
        if column in df.columns:
            # Assuming the metric is in [0,1], convert to percentage
            value = df[column].mean() * 100
            data.append({'Model': method, metric.capitalize(): value})
        else:
            # Handle missing columns by assigning NaN or a default value
            data.append({'Model': method, metric.capitalize(): 0.0})

    metric_df = pd.DataFrame(data)

    # Ensure the DataFrame follows the methods order
    metric_df['Model'] = pd.Categorical(metric_df['Model'], categories=methods, ordered=True)
    metric_df = metric_df.sort_values('Model')

    # Initialize the matplotlib figure
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Create the bar plot with consistent order
    bar_plot = sns.barplot(
        x='Model',
        y=metric.capitalize(),
        data=metric_df,
        palette='viridis',
        order=methods
    )

    # Set y-axis from 0 to 100
    plt.ylim(0, 100)

    # Add value labels above each bar
    for p in bar_plot.patches:
        height = p.get_height()
        bar_plot.annotate(
            f'{height:.1f}%',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=10, color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )

    # Set titles and labels
    plt.title(f'{metric.capitalize()} by OCR Model', fontsize=18)
    plt.ylabel(f'{metric.capitalize()} (%)', fontsize=14)
    plt.xlabel('OCR Model', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save the plot
    filename = f'bar_chart_{metric}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    logger.info(f"Saved bar chart for '{metric}' as '{filename}'.")

# =======================
# Histogram Functions
# =======================

def generate_and_save_histogram_data(df: pd.DataFrame, methods: List[str], metrics: List[str], output_dir: str) -> None:
    """
    Generates histogram data with 10% bin sizes for each model and metric and saves them as CSV files.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the metrics.
        methods (List[str]): List of OCR model names.
        metrics (List[str]): List of metrics to generate histograms for.
        output_dir (str): Directory where histogram data CSVs will be saved.
    """
    bin_edges = [i for i in range(0, 101, 10)]  # 0,10,20,...,100
    
    for metric in metrics:
        histogram_data = {'Model': methods}
        for method in methods:
            column = f'{method}_{metric}'
            if column in df.columns:
                # Convert metric to percentage
                data = df[column] * 100
                counts = pd.cut(data, bins=bin_edges, right=False, include_lowest=True).value_counts().sort_index()
                histogram_data[method] = counts.values
            else:
                histogram_data[method] = [0] * (len(bin_edges) - 1)
        
        histogram_df = pd.DataFrame(histogram_data)
        histogram_df.to_csv(os.path.join(output_dir, f'histogram_data_{metric}.csv'), index=False)
        logger.info(f"Saved histogram data for '{metric}' as 'histogram_data_{metric}.csv'.")

def plot_histograms(methods: List[str], metrics: List[str], output_dir_data: str, output_dir_plots: str) -> None:
    """
    Plots histograms based on the generated histogram data with 10% bin sizes.
    
    Args:
        methods (List[str]): List of OCR model names.
        metrics (List[str]): List of metrics to plot histograms for.
        output_dir_data (str): Directory where histogram data CSVs are saved.
        output_dir_plots (str): Directory where histogram plots will be saved.
    """
    bin_labels = [f'{i}-{i+10}%' for i in range(0, 100, 10)]
    bin_centers = [i + 5 for i in range(0, 100, 10)]
    
    for metric in metrics:
        data_path = os.path.join(output_dir_data, f'histogram_data_{metric}.csv')
        if not os.path.exists(data_path):
            logger.warning(f"Histogram data for '{metric}' not found at '{data_path}'. Skipping plot.")
            continue
        
        histogram_df = pd.read_csv(data_path)
        plt.figure(figsize=(14, 8))
        
        # Plot each model's histogram
        for idx, method in enumerate(methods):
            if method in histogram_df.columns:
                counts = histogram_df[method]
                # Offset bars for each model to prevent complete overlap
                offset = (idx - len(methods)/2) * 0.8  # Adjust offset based on number of methods
                plt.bar(
                    [x + offset for x in bin_centers],
                    counts,
                    width=8,
                    alpha=0.6,
                    label=method
                )
        
        plt.title(f'Histogram of {metric.replace("_", " ").title()}')
        plt.xlabel('Metric (%)')
        plt.ylabel('Count')
        plt.xticks(bin_centers, bin_labels)
        plt.legend(title="OCR Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f'histogram_{metric}.png'
        plt.savefig(os.path.join(output_dir_plots, filename))
        plt.close()
        logger.info(f"Saved histogram plot for '{metric}' as '{filename}'.")

# =======================
# Main Function
# =======================

def main(config: Dict[str, Any]) -> None:
    logger.info("Starting processing...")
    
    # Read primary and BART CSV files
    df_primary = read_csv(config["files"]["primary_input"], config["separators"]["primary_input"])
    display_df_info(df_primary, config["files"]["primary_input"])
    
    df_bart = read_csv(config["files"]["bart_input"], config["separators"]["bart_input"])
    display_df_info(df_bart, config["files"]["bart_input"])
    
    # Merge DataFrames on 'id'
    logger.info("Merging primary and BART DataFrames on 'id'...")
    try:
        df = pd.merge(df_primary, df_bart, on='id', how='inner', suffixes=('_x', '_y'))
        logger.info(f"Merged DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
        display_df_info(df, "Merged DataFrame")
    except Exception as e:
        logger.error(f"Error merging DataFrames: {e}")
        raise
    
    # Drop redundant '_y' columns
    y_columns = [col for col in df.columns if col.endswith('_y')]
    if y_columns:
        df.drop(columns=y_columns, inplace=True)
        logger.info(f"Dropped redundant '_y' columns: {y_columns}")
        display_df_info(df, "DataFrame After Dropping '_y' Columns")
    
    # Rename '_x' columns to remove suffix
    x_columns = [col for col in df.columns if col.endswith('_x')]
    rename_mapping = {col: col[:-2] for col in x_columns}
    if rename_mapping:
        df.rename(columns=rename_mapping, inplace=True)
        logger.info(f"Renamed '_x' columns: {rename_mapping}")
        display_df_info(df, "DataFrame After Renaming Columns")
    
    # Normalize 'transcription' as ground truth
    if 'transcription' in df.columns:
        logger.info("Normalizing 'transcription' column as 'normalized_transcription'...")
        df['normalized_transcription'] = df['transcription'].progress_apply(
            lambda x: normalize_text(
                x, 
                config["text_processing"]["max_char_length"], 
                config["text_processing"]["phrases_to_delete"]
            )
        )
        logger.info("Normalization of 'transcription' completed.\n")
    else:
        logger.error("Error: 'transcription' column not found in DataFrame.")
        raise KeyError("'transcription' column not found.")
    
    # Drop rows after first 1000 if debugging flag is set
    if config["debugging"]["drop_after_1000"]:
        logger.info("Dropping rows after the first 1000 for debugging purposes.")
        df = df.iloc[:1000]
        logger.info(f"DataFrame now has {df.shape[0]} rows.\n")
        display_df_info(df, "DataFrame After Dropping Rows")
    
    # Fill NaN values with empty strings
    df.fillna('', inplace=True)
    
    # Normalize OCR method columns
    logger.info("Normalizing OCR method transcription columns...")
    for method in config["transcription_columns"]:
        norm_col = f'normalized_{method}'
        logger.info(f"  - Normalizing '{method}' -> '{norm_col}'")
        df[norm_col] = df[method].progress_apply(
            lambda x: normalize_text(
                x, 
                config["text_processing"]["max_char_length"], 
                config["text_processing"]["phrases_to_delete"]
            )
        )
    logger.info("Normalization of OCR methods completed.\n")
    
    # Ensure normalized columns are within max_char_length
    logger.info("Ensuring all normalized transcriptions are limited to maximum character length...")
    df = ensure_length(
        df,
        methods=config["transcription_columns"],
        max_chars=config["text_processing"]["max_char_length"]
    )
    logger.info("Length enforcement completed.\n")
    
    # Flag repeated_text
    logger.info("Flagging rows with excessive repetition as 'repeated_text'...")
    df['repeated_text'] = df.progress_apply(flag_repeated_text, axis=1)
    logger.info("Flagging completed.\n")
    
    # Compare texts and compute similarities
    logger.info("Comparing texts and computing similarities...")
    for method in config["transcription_columns"]:
        logger.info(f"  - Processing comparisons for: {method}")
        df = pd.concat(
            [df, df.progress_apply(
                lambda row: compare_texts(
                    row,
                    method,
                    config["thresholds"],
                    config["text_processing"]["max_diff_length"]
                ),
                axis=1
            )],
            axis=1
        )
    logger.info("Text comparisons and similarity computations completed.\n")
    
    # Save the updated DataFrame
    save_csv(df, config["files"]["output"], config["separators"]["output"])
    
    # Create a sample CSV
    process_sample(
        df,
        sample_size=config["sample"]["size"],
        output_path=config["files"]["output_sample"],
        separator=config["separators"]["output"]
    )
    
    # Create output directories
    create_output_dirs([
        config["output_dirs"]["plots"],
        config["output_dirs"]["summary_statistics"],
        config["output_dirs"]["histogram_data"],
        config["output_dirs"]["histogram_plots"]
    ])
    
    # Generate overlayed histograms
    logger.info("Generating summary overlayed plots with median annotations...")
    for metric in config["metrics"]:
        plot_overlayed_histograms(
            df,
            methods=config["transcription_columns"],
            metric=metric,
            output_dir=config["output_dirs"]["plots"]
        )
    logger.info("Overlayed summary plots generated and saved.\n")
    
    # Generate bar charts
    logger.info("Generating bar charts for each metric...")
    for metric in config["metrics"]:
        plot_bar_chart(
            df,
            methods=config["transcription_columns"],  # Ensures consistent order
            metric=metric,
            output_dir=config["output_dirs"]["plots"]
        )
    logger.info("Bar charts generated and saved.\n")
    
    # Generate top 3 and bottom 3 model plots
    logger.info("Generating top 3 and bottom 3 model plots...")
    for metric in config["metrics"]:
        plot_top_bottom_models(
            df,
            methods=config["transcription_columns"],
            metric=metric,
            output_dir=config["output_dirs"]["plots"]
        )
    logger.info("Top 3 and bottom 3 model plots generated and saved.\n")
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    generate_summary_statistics(
        df,
        methods=config["transcription_columns"],
        metrics=config["metrics"],
        output_dir=config["output_dirs"]["summary_statistics"]
    )
    logger.info("Summary statistics generated and saved.\n")
    
    # Generate histogram data and plots
    logger.info("Generating histogram data and plots...")
    generate_and_save_histogram_data(
        df,
        methods=config["transcription_columns"],
        metrics=config["metrics"],
        output_dir=config["output_dirs"]["histogram_data"]
    )
    plot_histograms(
        methods=config["transcription_columns"],
        metrics=config["metrics"],
        output_dir_data=config["output_dirs"]["histogram_data"],
        output_dir_plots=config["output_dirs"]["histogram_plots"]
    )
    logger.info("Histogram data and plots generated and saved.\n")
    
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    import nltk
    nltk.download('punkt')
    
    main(CONFIG)
