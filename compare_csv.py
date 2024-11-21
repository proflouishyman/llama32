import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher, ndiff
from fuzzywuzzy import fuzz
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# Parameters and Variables
# =======================

# Input and output filenames
PRIMARY_INPUT_FILE = 'truncated_csv.csv'                # Primary input CSV file
BART_INPUT_FILE = 'processed_1000_testing_csv.csv'      # Additional BART CSV file
OUTPUT_FILE = 'updated_file.csv'                        # Output CSV file with results
OUTPUT_SAMPLE_FILE = 'sample_updated_file.csv'          # Sample output CSV file with 10 rows

# CSV file separators
INPUT_SEPARATOR = ','       # Primary input CSV is comma-delimited
BART_SEPARATOR = ','        # BART input CSV delimiter
OUTPUT_SEPARATOR = '|'     # Desired output delimiter

# Similarity thresholds
SIMILARITY_THRESHOLD_DIFFLIB = 0.93  
SIMILARITY_THRESHOLD_FUZZYWUZZY = 93  # FuzzyWuzzy uses 0-100 scale
SIMILARITY_THRESHOLD_JACCARD = 0.93    # Example threshold for Jaccard

# List of transcription columns to normalize and compare
# Excluding 'transcription' from analysis and plotting
TRANSCRIPTION_COLUMNS = [
    'pyte_ocr', 'chatgpt_ocr', 'LLAMA32_BASE',
    'BART_untuned', 'BART_gold_100', 'BART_gold_1000', 'BART_gold_10000',
    'BART_silver_100', 'BART_silver_1000', 'BART_silver_10000'
]

# Maximum characters per transcription to estimate ~1000 tokens (assuming ~5 chars per token)
MAX_CHAR_LENGTH = 5000

# Maximum length of differences string (to prevent excessively long entries)
MAX_DIFF_LENGTH = 1000  # Adjust as needed

# Subsample size for sample CSV
SAMPLE_SIZE = 10

# Flag to drop rows after first 1000 for debugging
DROP_AFTER_1000 = True

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
    equal_jaccard = similarity_jaccard >= SIMILARITY_THRESHOLD_JACCARD
    
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
    Excludes 'transcription' from plotting.
    """
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    for method, color in zip(methods, colors):
        column_name = f'{method}_{metric}'
        if column_name in df.columns:
            sns.histplot(df[column_name], bins=30, kde=True, label=method, color=color, stat="density", common_norm=False, alpha=0.5)
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
    Ensure all normalized transcription columns are limited to max_chars.
    """
    for col in TRANSCRIPTION_COLUMNS:
        norm_col = f'normalized_{col}'
        df[norm_col] = df[norm_col].apply(lambda x: x if len(x) <= max_chars else x[:max_chars] + '... [truncated]')
    return df

def display_csv_info(df, name):
    """
    Display basic information about the DataFrame.
    """
    print(f"--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes}\n")

def generate_summary_statistics(df, methods, metrics, output_dir):
    """
    Generate and save summary statistics for each measurement and model.
    """
    summary = {}
    for metric in metrics:
        summary[metric] = {}
        for method in methods:
            column = f'{method}_{metric}'
            if column in df.columns:
                summary[metric][method] = {
                    'Mean': df[column].mean(),
                    'Median': df[column].median(),
                    'Std Dev': df[column].std(),
                    'Min': df[column].min(),
                    'Max': df[column].max(),
                    '25%': df[column].quantile(0.25),
                    '75%': df[column].quantile(0.75)
                }
    # Convert to DataFrame for better visualization
    for metric in metrics:
        metric_df = pd.DataFrame(summary[metric]).T
        metric_df = metric_df[['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25%', '75%']]
        # Save as CSV
        metric_csv = os.path.join(output_dir, f'summary_statistics_{metric}.csv')
        metric_df.to_csv(metric_csv)
        print(f"Saved summary statistics for {metric} as '{metric_csv}'.")
        
        # Optionally, print to console
        print(f"\nSummary Statistics for {metric}:\n{metric_df}\n")

# =======================
# Main Processing
# =======================

if __name__ == "__main__":
    print("Starting processing...")
    
    # Read the primary CSV file
    print(f"Reading primary CSV file: {PRIMARY_INPUT_FILE}")
    try:
        df_primary = pd.read_csv(PRIMARY_INPUT_FILE, sep=INPUT_SEPARATOR, dtype=str)  # Read all columns as strings
        print("Primary CSV file read successfully.")
        display_csv_info(df_primary, PRIMARY_INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: The file {PRIMARY_INPUT_FILE} was not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading the primary CSV file: {e}")
        exit(1)
    
    # Read the BART CSV file
    print(f"Reading BART CSV file: {BART_INPUT_FILE}")
    try:
        df_bart = pd.read_csv(BART_INPUT_FILE, sep=BART_SEPARATOR, dtype=str)  # Read all columns as strings
        print("BART CSV file read successfully.")
        display_csv_info(df_bart, BART_INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: The file {BART_INPUT_FILE} was not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading the BART CSV file: {e}")
        exit(1)
    
    # Merge the two DataFrames on 'id'
    print("Merging primary and BART DataFrames on 'id'...")
    try:
        df = pd.merge(df_primary, df_bart, on='id', how='inner', suffixes=('_x', '_y'))
        print(f"Merged DataFrame has {len(df)} rows and {len(df.columns)} columns.\n")
        display_csv_info(df, "Merged DataFrame")
    except Exception as e:
        print(f"Error merging DataFrames: {e}")
        exit(1)
    
    # Drop redundant '_y' columns as they are identical
    print("Dropping redundant '_y' columns...")
    y_columns = [col for col in df.columns if col.endswith('_y')]
    df.drop(columns=y_columns, inplace=True)
    print(f"Dropped columns: {y_columns}\n")
    display_csv_info(df, "DataFrame After Dropping '_y' Columns")
    
    # Rename '_x' columns to remove suffix
    print("Renaming '_x' columns to remove suffix...")
    x_columns = [col for col in df.columns if col.endswith('_x')]
    rename_mapping = {col: col[:-2] for col in x_columns}  # Remove '_x' suffix
    df.rename(columns=rename_mapping, inplace=True)
    print(f"Renamed columns: {rename_mapping}\n")
    display_csv_info(df, "DataFrame After Renaming Columns")
    
    # Normalize the 'transcription' column separately as ground truth
    print("Normalizing 'transcription' column as 'normalized_transcription'...")
    if 'transcription' in df.columns:
        df['normalized_transcription'] = df['transcription'].progress_apply(lambda x: normalize_text(x, max_chars=MAX_CHAR_LENGTH))
        print("Normalization of 'transcription' completed.\n")
    else:
        print("Error: 'transcription' column not found in DataFrame.")
        exit(1)
    
    # Update TRANSCRIPTION_COLUMNS to reflect renamed columns, excluding 'transcription'
    TRANSCRIPTION_COLUMNS = [
        'pyte_ocr', 'chatgpt_ocr', 'LLAMA32_BASE',
        'BART_untuned', 'BART_gold_100', 'BART_gold_1000', 'BART_gold_10000',
        'BART_silver_100', 'BART_silver_1000', 'BART_silver_10000'
    ]
    
    # Drop rows after first 1000 if flag is set
    if DROP_AFTER_1000:
        print("Dropping rows after the first 1000 for debugging purposes.")
        df = df.iloc[:1000]
        print(f"DataFrame now has {len(df)} rows.\n")
        display_csv_info(df, "DataFrame After Dropping Rows")
    
    # Fill NaN values with empty strings to avoid errors during normalization
    df.fillna('', inplace=True)
    
    # Normalize the texts and create new 'normalized_' columns
    print("Normalizing transcription columns...")
    for col in TRANSCRIPTION_COLUMNS:
        norm_col = f'normalized_{col}'
        print(f"  - Normalizing column: {col} -> {norm_col}")
        df[norm_col] = df[col].progress_apply(lambda x: normalize_text(x, max_chars=MAX_CHAR_LENGTH))
    print("Normalization completed.\n")
    
    # Ensure all normalized transcriptions are limited to max_chars
    print("Ensuring all normalized transcriptions are limited to maximum character length...")
    df = ensure_length(df, max_chars=MAX_CHAR_LENGTH)
    print("Length enforcement completed.\n")
    
    # Apply comparison for each method and store results
    print("Comparing texts and computing similarities...")
    for method_col in TRANSCRIPTION_COLUMNS:
        print(f"  - Processing comparisons for: {method_col}")
        results = df.progress_apply(compare_texts, axis=1, method_col=method_col)
        df = pd.concat([df, results], axis=1)
    print("Text comparisons and similarity computations completed.\n")
    
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
        print(f"Main CSV file saved successfully as '{OUTPUT_FILE}'.\n")
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
        print(f"Sample CSV file saved successfully as '{OUTPUT_SAMPLE_FILE}'.\n")
    except Exception as e:
        print(f"Error saving the sample CSV file: {e}")
        exit(1)
    
    # Define output directories for plots and summary statistics
    PLOT_OUTPUT_DIR = 'plots'
    SUMMARY_OUTPUT_DIR = 'summary_statistics'
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
    
    # Generate summary plots overlaying results of different approaches and models
    print("Generating summary overlayed plots...")
    
    # Metrics to plot
    metrics_to_plot = ['similarity_difflib', 'fuzz_ratio', 'jaccard_similarity']
    
    # Overlayed histograms for each metric
    for metric in metrics_to_plot:
        plot_overlayed_histograms(df, TRANSCRIPTION_COLUMNS, metric, PLOT_OUTPUT_DIR)
    
    print("Overlayed summary plots generated and saved.\n")
    
    # Generate and save summary statistics
    print("Generating summary statistics...")
    generate_summary_statistics(df, TRANSCRIPTION_COLUMNS, metrics_to_plot, SUMMARY_OUTPUT_DIR)
    print("Summary statistics generated and saved.\n")
    
    print("Processing completed successfully.")
