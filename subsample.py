import pandas as pd
import csv  # Ensure the csv module is imported

# -----------------------------
# Parameters and File Paths
# -----------------------------
INPUT_CSV = 'updated_file.csv'       # Replace with your actual CSV file path
OUTPUT_SAMPLE_CSV = 'sample_file.csv'  # Desired output CSV file for the subsample
CSV_SEPARATOR = ','                  # Ensure this matches your CSV delimiter
SAMPLE_SIZE = 100                    # Number of lines to extract

# -----------------------------
# Processing
# -----------------------------
try:
    # Read the first 100 rows from the CSV
    df_sample = pd.read_csv(
        INPUT_CSV,
        sep=CSV_SEPARATOR,
        engine='python',            # Python engine handles complex CSVs better
        quoting=csv.QUOTE_ALL,      # Ensures all fields are quoted
        escapechar='\\',            # Defines escape character if needed
        nrows=SAMPLE_SIZE,          # Number of rows to read
        dtype=str,                  # Read all columns as strings to prevent type issues
        encoding='utf-8',           # Ensure UTF-8 encoding
        on_bad_lines='skip'         # Skip lines with parsing errors
    )
    print(f"Successfully read the first {SAMPLE_SIZE} lines.")
    
    # Save the subsample to a new CSV file
    df_sample.to_csv(
        OUTPUT_SAMPLE_CSV,
        index=False,
        sep=CSV_SEPARATOR,
        quoting=csv.QUOTE_ALL,       # Enclose all fields in quotes
        escapechar='\\',
        quotechar='"',
        doublequote=True,
        encoding='utf-8'
    )
    print(f"Sample CSV saved as '{OUTPUT_SAMPLE_CSV}'.")
    
except FileNotFoundError:
    print(f"Error: The file '{INPUT_CSV}' was not found.")
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
