import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher, ndiff
from fuzzywuzzy import fuzz
from tqdm import tqdm
import csv  # Importing the csv module for quoting constants


#Processes CSV to measure the differences in OCR outputs

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

# Subsample size
SAMPLE_SIZE = 10

# Initialize tqdm for pandas apply functions
tqdm.pandas(desc="Progress")

# =======================
# Function Definitions
# =======================

def normalize_text(text):
    """
    Normalize text by lowercasing, removing accents, punctuation, and extra whitespace.
    """
    if not isinstance(text, str):
        return ''
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
    Computes similarities using difflib and FuzzyWuzzy.
    Returns a Series with comparison results.
    """
    human_text = row['normalized_transcription']
    method_text = row[f'normalized_{method_col}']
    
    # Compute difflib similarity
    similarity_difflib = compute_similarity_difflib(human_text, method_text)
    equal_difflib = similarity_difflib >= SIMILARITY_THRESHOLD_DIFFLIB
    
    # Compute FuzzyWuzzy similarities
    similarity_fuzzy = compute_similarity_fuzzywuzzy(human_text, method_text)
    # Determine equality based on FuzzyWuzzy's ratio
    equal_fuzzy = similarity_fuzzy['fuzz_ratio'] >= SIMILARITY_THRESHOLD_FUZZYWUZZY
    
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
        f'{method_col}_differences': differences
    })

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
        df[norm_col] = df[col].progress_apply(normalize_text)
    
    # Apply comparison for each method and store results
    print("Comparing texts and computing similarities...")
    for method_col in TRANSCRIPTION_COLUMNS[1:]:  # Skip 'transcription' column
        print(f"  - Processing comparisons for: {method_col}")
        results = df.progress_apply(compare_texts, axis=1, method_col=method_col)
        df = pd.concat([df, results], axis=1)
    
    # Save the updated DataFrame to the main output CSV file with '||' as delimiter
    print(f"Saving updated DataFrame to CSV: {OUTPUT_FILE}")
    try:
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            sep=OUTPUT_SEPARATOR,          # Change delimiter to '||'
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
        
        # Save the subsample to a new CSV file with '||' as delimiter
        df_sample.to_csv(
            OUTPUT_SAMPLE_FILE,
            index=False,
            sep=OUTPUT_SEPARATOR,          # Use the same delimiter '||'
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
    
    print("Processing completed successfully.")
