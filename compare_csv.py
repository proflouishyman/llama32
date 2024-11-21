import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher, ndiff
from fuzzywuzzy import fuzz
from tqdm import tqdm
import csv  # Importing the csv module for quoting constants
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# Parameters and Variables
# =======================

# Input and output filenames
INPUT_FILE = 'truncated_csv.csv'          # Replace with your actual input CSV file
OUTPUT_FILE = 'updated_file.csv'          # Output CSV file with results
OUTPUT_SAMPLE_FILE = 'sample_updated_file.csv'  # Sample output CSV file with 10 rows

# CSV file separators
INPUT_SEPARATOR = ','       # Input CSV is comma-delimited
OUTPUT_SEPARATOR = '|'     # Desired output delimiter

# Similarity thresholds (optional)
SIMILARITY_THRESHOLD_DIFFLIB = 0.95   
SIMILARITY_THRESHOLD_FUZZYWUZZY = 95  # FuzzyWuzzy uses 0-100 scale

# List of columns containing transcriptions to normalize and compare
TRANSCRIPTION_COLUMNS = ['transcription', 'pyte_ocr', 'chatgpt_ocr', 'LLAMA32_BASE']

# Maximum length of differences string (to prevent excessively long entries)
MAX_DIFF_LENGTH = 1000  # Adjust as needed

# Maximum characters per transcription to estimate ~1000 tokens (assuming ~5 chars per token)
MAX_CHAR_LENGTH = 5000

# Subsample size
SAMPLE_SIZE = 10

# Initialize tqdm for pandas apply functions
tqdm.pandas(desc="Progress")

# =======================
# Function Definitions
# =======================

def normalize_text(text, max_chars=MAX_CHAR_LENGTH):
    """
    Normalize text by:
    - Lowercasing
    - Removing accents and diacritics
    - Removing punctuation
    - Removing extra whitespace
    - Truncating or padding to a maximum number of characters
    """
    if not isinstance(text, str):
        return ' ' * max_chars  # Pad with spaces if not a string
    
    # Normalize Unicode characters to NFC form
    text = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents and diacritics
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace any whitespace character with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    text = text.strip()
    
    # Replace newlines with spaces to prevent issues in CSV
    text = text.replace('\n', ' ')
    
    # Truncate to max_chars
    if len(text) > max_chars:
        text = text[:max_chars - 15] + '... [truncated]'
    else:
        # Pad with spaces to reach max_chars
        text = text.ljust(max_chars)
    
    return text

def compute_similarity_difflib(text1, text2):
    """
    Compute similarity ratio using difflib's SequenceMatcher.
    Returns a float between 0 and 1.
    """
    return SequenceMatcher(None, text1, text2).ratio()

def compute_similarity_fuzzywuzzy(text1, text2):
    """
    Compute similarity ratios using FuzzyWuzzy's different methods.
    Returns a dictionary with various similarity scores.
    """
    return {
        'fuzz_ratio': fuzz.ratio(text1, text2),
        'fuzz_partial_ratio': fuzz.partial_ratio(text1, text2),
        'fuzz_token_sort_ratio': fuzz.token_sort_ratio(text1, text2),
        'fuzz_token_set_ratio': fuzz.token_set_ratio(text1, text2)
    }

def compute_jaccard_similarity(text1, text2):
    """
    Compute Jaccard similarity index between two texts.
    Returns a float between 0 and 1.
    """
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def get_differences(text1, text2, max_length=MAX_DIFF_LENGTH):
    """
    Generate a string showing differences between two texts using difflib's ndiff.
    """
    # Split texts into words
    words1 = text1.split()
    words2 = text2.split()
    # Generate diff using ndiff
    diff = list(ndiff(words1, words2))
    # Filter out unchanged words
    changes = [d for d in diff if d.startswith('- ') or d.startswith('+ ')]
    # Join changes into a single string
    differences = ' '.join(changes)
    # Truncate if necessary
    if len(differences) > max_length:
        differences = differences[:max_length] + '... [truncated]'
    return differences

def compare_texts(row, method_col):
    """
    Compare the normalized human transcription with a specific method's transcription.
    Computes similarities using difflib, FuzzyWuzzy, and Jaccard similarity.
    Returns a Series with comparison results.
    """
    human_text = row['normalized_transcription'].strip()
    method_text = row[f'normalized_{method_col}'].strip()
    
    # Compute difflib similarity
    similarity_difflib = compute_similarity_difflib(human_text, method_text)
    equal_difflib = similarity_difflib >= SIMILARITY_THRESHOLD_DIFFLIB
    
    # Compute FuzzyWuzzy similarities
    similarity_fuzzy = compute_similarity_fuzzywuzzy(human_text, method_text)
    # Determine equality based on FuzzyWuzzy's ratio
    equal_fuzzy = similarity_fuzzy['fuzz_ratio'] >= SIMILARITY_THRESHOLD_FUZZYWUZZY
    
    # Compute Jaccard similarity
    similarity_jaccard = compute_jaccard_similarity(human_text, method_text)
    equal_jaccard = similarity_jaccard >= 0.8  # Example threshold for Jaccard
    
    # Get differences
    differences = get_differences(human_text, method_text)
    
    # Compile results
    return pd.Series({
        f'{method_col}_equal_difflib': equal_difflib,
        f'{method_col}_similarity_difflib': similarity_difflib,
        f'{method_col}_fuzz_ratio': similarity_fuzzy['fuzz_ratio'],
        f'{method_col}_fuzz_partial_ratio': similarity_fuzzy['fuzz_partial_ratio'],
        f'{method_col}_fuzz_token_sort_ratio': similarity_fuzzy['fuzz_token_sort_ratio'],
        f'{method_col}_fuzz_token_set_ratio': similarity_fuzzy['fuzz_token_set_ratio'],
        f'{method_col}_equal_fuzzywuzzy': equal_fuzzy,
        f'{method_col}_jaccard_similarity': similarity_jaccard,
        f'{method_col}_equal_jaccard': equal_jaccard,
        f'{method_col}_differences': differences
    })

def plot_overlayed_histograms(df, methods, metric, output_dir):
    """
    Plot overlayed histograms for a given metric across different OCR methods.
    """
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    for method, color in zip(methods, colors):
        sns.histplot(df[f'{method}_{metric}'], bins=30, kde=True, label=method, color=color, stat="density", common_norm=False, alpha=0.5)
    plt.title(f'Overlayed Histogram of {metric.replace("_", " ").title()}')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    filename = f'overlayed_histogram_{metric}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved overlayed histogram for {metric} as '{filename}'.")

def ensure_length(df, max_chars=MAX_CHAR_LENGTH):
    """
    Ensure all transcription columns are limited to max_chars.
    """
    for col in TRANSCRIPTION_COLUMNS:
        norm_col = f'normalized_{col}'
        df[norm_col] = df[norm_col].apply(lambda x: x if len(x) <= max_chars else x[:max_chars] + '... [truncated]')
    return df

# =======================
# Main Processing
# =======================

if __name__ == "__main__":
    print("Starting processing...")
    
    # Read the CSV file
    print(f"Reading CSV file: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, sep=INPUT_SEPARATOR, dtype=str)  # Read all columns as strings
        print("CSV file read successfully.")
    except FileNotFoundError:
        print(f"Error: The file {INPUT_FILE} was not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading the file: {e}")
        exit(1)
    
    # Fill NaN values with empty strings to avoid errors during normalization
    df.fillna('', inplace=True)
    
    # Normalize the texts and create new columns
    print("Normalizing transcription columns...")
    for col in TRANSCRIPTION_COLUMNS:
        norm_col = f'normalized_{col}'
        print(f"  - Normalizing column: {col}")
        df[norm_col] = df[col].progress_apply(lambda x: normalize_text(x, max_chars=MAX_CHAR_LENGTH))
    
    # Ensure all normalized transcriptions are limited to max_chars
    print("Ensuring all normalized transcriptions are limited to maximum character length...")
    df = ensure_length(df, max_chars=MAX_CHAR_LENGTH)
    
    # Apply comparison for each method and store results
    print("Comparing texts and computing similarities...")
    for method_col in TRANSCRIPTION_COLUMNS[1:]:  # Skip 'transcription' column
        print(f"  - Processing comparisons for: {method_col}")
        results = df.progress_apply(compare_texts, axis=1, method_col=method_col)
        df = pd.concat([df, results], axis=1)
    
    # Save the updated DataFrame to the main output CSV file with '|' as delimiter
    print(f"Saving updated DataFrame to CSV: {OUTPUT_FILE}")
    try:
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            sep=OUTPUT_SEPARATOR,          # Change delimiter to '|'
            quoting=csv.QUOTE_ALL,         # Enclose all fields in quotes
            escapechar='\\',               # Define escape character if needed
            quotechar='"',                 # Define quote character
            doublequote=True,              # Handle double quotes by doubling them
            encoding='utf-8'               # Ensure UTF-8 encoding
        )
        print(f"Main CSV file saved successfully as '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"Error saving the main CSV file: {e}")
        exit(1)
    
    # Create a subsample of the first 10 rows
    print(f"Creating a subsample CSV with the first {SAMPLE_SIZE} rows: {OUTPUT_SAMPLE_FILE}")
    try:
        # Extract the first SAMPLE_SIZE rows
        df_sample = df.head(SAMPLE_SIZE)
        
        # Save the subsample to a new CSV file with '|' as delimiter
        df_sample.to_csv(
            OUTPUT_SAMPLE_FILE,
            index=False,
            sep=OUTPUT_SEPARATOR,          # Use the same delimiter '|'
            quoting=csv.QUOTE_ALL,         # Enclose all fields in quotes
            escapechar='\\',               # Define escape character if needed
            quotechar='"',                 # Define quote character
            doublequote=True,              # Handle double quotes by doubling them
            encoding='utf-8'               # Ensure UTF-8 encoding
        )
        print(f"Sample CSV file saved successfully as '{OUTPUT_SAMPLE_FILE}'.")
    except Exception as e:
        print(f"Error saving the sample CSV file: {e}")
        exit(1)
    
    # Generate summary plots overlaying results of different approaches and models
    print("Generating summary overlayed plots...")
    
    # Overlayed histograms for difflib similarity
    plot_overlayed_histograms(df, TRANSCRIPTION_COLUMNS[1:], 'similarity_difflib', OUTPUT_SEPARATOR)
    
    # Overlayed histograms for FuzzyWuzzy ratio
    plot_overlayed_histograms(df, TRANSCRIPTION_COLUMNS[1:], 'fuzz_ratio', OUTPUT_SEPARATOR)
    
    # Overlayed histograms for Jaccard similarity
    plot_overlayed_histograms(df, TRANSCRIPTION_COLUMNS[1:], 'jaccard_similarity', OUTPUT_SEPARATOR)
    
    print("Processing completed successfully.")
