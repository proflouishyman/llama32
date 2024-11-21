import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import logging

# =======================
# Parameters and Variables
# =======================

# Input and output filenames
INPUT_FILE = 'updated_file.csv'          # Replace with your actual input CSV file
OUTPUT_DIRECTORY = 'analysis_outputs'     # Directory to save graphs, summaries, and logs

# CSV file separators
INPUT_SEPARATOR = '|'  # Ensure this matches the delimiter used in writing the CSV

# List of OCR methods to analyze
OCR_METHODS = ['pyte_ocr', 'chatgpt_ocr', 'LLAMA32_BASE']

# Log file configuration
LOG_FILE = os.path.join(OUTPUT_DIRECTORY, 'analysis_log.log')

# =======================
# Function Definitions
# =======================

def create_output_directory(directory):
    """
    Create the output directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def setup_logging(log_path):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized.")

def read_csv_file(file_path, separator):
    """
    Read the CSV file with the specified separator.
    """
    try:
        df = pd.read_csv(
            file_path,
            sep=separator,
            engine='python',
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            dtype=str,
            encoding='utf-8',
            on_bad_lines='skip'  # Skips lines with parsing errors
        )
        logging.info(f"Successfully read the file: {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"The file '{file_path}' was not found.")
        print(f"Error: The file '{file_path}' was not found.")
        exit(1)
    except pd.errors.ParserError as e:
        logging.error(f"ParserError: {e}")
        print(f"ParserError: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the file: {e}")
        print(f"An unexpected error occurred while reading the file: {e}")
        exit(1)

def compute_summary_statistics(df, methods, include_nan=True):
    """
    Compute summary statistics for each OCR method.
    If include_nan is False, exclude rows with NaN in similarity or fuzz_ratio columns.
    """
    summary = {}
    for method in methods:
        similarity_difflib = f'{method}_similarity_difflib'
        fuzz_ratio = f'{method}_fuzz_ratio'
        equal_difflib = f'{method}_equal_difflib'
        equal_fuzzywuzzy = f'{method}_equal_fuzzywuzzy'

        # Convert to numeric, coerce errors to NaN
        df[similarity_difflib] = pd.to_numeric(df[similarity_difflib], errors='coerce')
        df[fuzz_ratio] = pd.to_numeric(df[fuzz_ratio], errors='coerce')

        if not include_nan:
            valid_df = df.dropna(subset=[similarity_difflib, fuzz_ratio])
            entry_count = len(valid_df)
            logging.info(f"Method '{method}': {entry_count} valid entries after excluding NaNs.")
        else:
            valid_df = df.copy()
            entry_count = len(valid_df)
            logging.info(f"Method '{method}': {entry_count} total entries including NaNs.")

        # Compute statistics
        summary[method] = {
            'Average Similarity (difflib)': valid_df[similarity_difflib].mean(),
            'Median Similarity (difflib)': valid_df[similarity_difflib].median(),
            'Average FuzzyWuzzy Ratio': valid_df[fuzz_ratio].mean(),
            'Median FuzzyWuzzy Ratio': valid_df[fuzz_ratio].median(),
            'Exact Matches (difflib)': valid_df[equal_difflib].astype(bool).sum(),
            'Exact Matches (FuzzyWuzzy)': valid_df[equal_fuzzywuzzy].astype(bool).sum(),
            'Total Entries': entry_count
        }

        # Log non-convertible entries
        non_numeric_sim = df[similarity_difflib].isna().sum()
        non_numeric_fuzz = df[fuzz_ratio].isna().sum()
        if non_numeric_sim > 0:
            logging.warning(f"{non_numeric_sim} entries in '{similarity_difflib}' could not be converted to float.")
        if non_numeric_fuzz > 0:
            logging.warning(f"{non_numeric_fuzz} entries in '{fuzz_ratio}' could not be converted to float.")

    summary_df = pd.DataFrame(summary).T  # Transpose for better readability
    return summary_df

def save_summary_statistics(summary_df, output_path):
    """
    Save the summary statistics DataFrame to a text file.
    """
    try:
        with open(output_path, 'w') as f:
            f.write("Summary Statistics:\n")
            f.write(summary_df.to_string())
        logging.info(f"Saved summary statistics to '{output_path}'.")
    except Exception as e:
        logging.error(f"Error saving summary statistics: {e}")
        print(f"Error saving summary statistics: {e}")

def plot_similarity_distributions(df, methods, output_dir, include_nan=True):
    """
    Plot histograms of similarity scores for each OCR method.
    """
    for method in methods:
        similarity_difflib = f'{method}_similarity_difflib'
        plt.figure(figsize=(8, 6))
        sns.histplot(df[similarity_difflib], bins=20, kde=True, color='skyblue')
        plt.title(f'Similarity Distribution (difflib) for {method} {"(Including NaN)" if include_nan else "(Excluding NaN)"}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        filename = f'{method}_similarity_difflib_histogram_{"incl_nan" if include_nan else "excl_nan"}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        logging.info(f"Saved histogram for {method} similarity_difflib with {'including' if include_nan else 'excluding'} NaN.")

def plot_fuzzywuzzy_ratios(df, methods, output_dir, include_nan=True):
    """
    Plot histograms of FuzzyWuzzy ratio scores for each OCR method.
    """
    for method in methods:
        fuzz_ratio = f'{method}_fuzz_ratio'
        plt.figure(figsize=(8, 6))
        sns.histplot(df[fuzz_ratio], bins=20, kde=True, color='salmon')
        plt.title(f'FuzzyWuzzy Ratio Distribution for {method} {"(Including NaN)" if include_nan else "(Excluding NaN)"}')
        plt.xlabel('FuzzyWuzzy Ratio')
        plt.ylabel('Frequency')
        plt.tight_layout()
        filename = f'{method}_fuzz_ratio_histogram_{"incl_nan" if include_nan else "excl_nan"}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        logging.info(f"Saved histogram for {method} fuzz_ratio with {'including' if include_nan else 'excluding'} NaN.")

def plot_exact_matches(summary_df_incl, summary_df_excl, methods, output_dir):
    """
    Plot bar charts showing the number of exact matches for each OCR method.
    """
    exact_difflib_incl = [summary_df_incl.loc[method, 'Exact Matches (difflib)'] for method in methods]
    exact_fuzzywuzzy_incl = [summary_df_incl.loc[method, 'Exact Matches (FuzzyWuzzy)'] for method in methods]

    exact_difflib_excl = [summary_df_excl.loc[method, 'Exact Matches (difflib)'] for method in methods]
    exact_fuzzywuzzy_excl = [summary_df_excl.loc[method, 'Exact Matches (FuzzyWuzzy)'] for method in methods]

    x = range(len(methods))
    width = 0.2  # Width of the bars

    plt.figure(figsize=(12, 6))
    plt.bar([p - width for p in x], exact_difflib_incl, width, label='Exact Matches (difflib) Incl NaN', color='green')
    plt.bar(x, exact_fuzzywuzzy_incl, width, label='Exact Matches (FuzzyWuzzy) Incl NaN', color='orange')
    plt.bar([p + width for p in x], exact_difflib_excl, width, label='Exact Matches (difflib) Excl NaN', color='lightgreen')
    plt.bar([p + 2*width for p in x], exact_fuzzywuzzy_excl, width, label='Exact Matches (FuzzyWuzzy) Excl NaN', color='peachpuff')

    plt.xlabel('OCR Methods')
    plt.ylabel('Number of Exact Matches')
    plt.title('Exact Matches Comparison by OCR Method')
    plt.xticks([p for p in x], methods)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exact_matches_comparison.png'))
    plt.close()
    logging.info("Saved exact matches comparison bar chart.")

def plot_similarity_boxplots(df_incl, df_excl, methods, output_dir):
    """
    Plot boxplots of similarity scores for each OCR method.
    """
    for method in methods:
        similarity_difflib = f'{method}_similarity_difflib'
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=[df_incl[similarity_difflib], df_excl[similarity_difflib]],
                    palette=['skyblue', 'lightgreen'])
        plt.xticks([0, 1], ['Including NaN', 'Excluding NaN'])
        plt.ylabel('Similarity Score (difflib)')
        plt.title(f'Similarity Score Boxplot for {method}')
        plt.tight_layout()
        filename = f'{method}_similarity_boxplot.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        logging.info(f"Saved similarity score boxplot for {method}.")

def plot_fuzzywuzzy_boxplots(df_incl, df_excl, methods, output_dir):
    """
    Plot boxplots of FuzzyWuzzy ratio scores for each OCR method.
    """
    for method in methods:
        fuzz_ratio = f'{method}_fuzz_ratio'
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=[df_incl[fuzz_ratio], df_excl[fuzz_ratio]],
                    palette=['salmon', 'lightcoral'])
        plt.xticks([0, 1], ['Including NaN', 'Excluding NaN'])
        plt.ylabel('FuzzyWuzzy Ratio')
        plt.title(f'FuzzyWuzzy Ratio Boxplot for {method}')
        plt.tight_layout()
        filename = f'{method}_fuzz_ratio_boxplot.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        logging.info(f"Saved FuzzyWuzzy ratio boxplot for {method}.")

# =======================
# Main Processing
# =======================

if __name__ == "__main__":
    print("Starting analysis...")

    # Create output directory
    create_output_directory(OUTPUT_DIRECTORY)

    # Set up logging
    setup_logging(LOG_FILE)

    # Read the main CSV file
    df = read_csv_file(INPUT_FILE, INPUT_SEPARATOR)

    # Compute summary statistics
    logging.info("Computing summary statistics including NaN...")
    summary_incl = compute_summary_statistics(df, OCR_METHODS, include_nan=True)
    logging.info("Computing summary statistics excluding NaN...")
    summary_excl = compute_summary_statistics(df, OCR_METHODS, include_nan=False)

    print("\nSummary Statistics (Including NaN):")
    print(summary_incl)
    print("\nSummary Statistics (Excluding NaN):")
    print(summary_excl)

    # Save summary statistics to text files
    save_summary_statistics(summary_incl, os.path.join(OUTPUT_DIRECTORY, 'summary_statistics_including_nan.txt'))
    save_summary_statistics(summary_excl, os.path.join(OUTPUT_DIRECTORY, 'summary_statistics_excluding_nan.txt'))

    # Generate and save visualizations
    plot_similarity_distributions(df, OCR_METHODS, OUTPUT_DIRECTORY, include_nan=True)
    plot_similarity_distributions(df, OCR_METHODS, OUTPUT_DIRECTORY, include_nan=False)

    plot_fuzzywuzzy_ratios(df, OCR_METHODS, OUTPUT_DIRECTORY, include_nan=True)
    plot_fuzzywuzzy_ratios(df, OCR_METHODS, OUTPUT_DIRECTORY, include_nan=False)

    plot_exact_matches(summary_incl, summary_excl, OCR_METHODS, OUTPUT_DIRECTORY)

    plot_similarity_boxplots(df, df.dropna(subset=[f'{method}_similarity_difflib' for method in OCR_METHODS] +
                                                [f'{method}_fuzz_ratio' for method in OCR_METHODS]),
                              OCR_METHODS, OUTPUT_DIRECTORY)

    plot_fuzzywuzzy_boxplots(df, df.dropna(subset=[f'{method}_similarity_difflib' for method in OCR_METHODS] +
                                                [f'{method}_fuzz_ratio' for method in OCR_METHODS]),
                              OCR_METHODS, OUTPUT_DIRECTORY)

    print("\nAnalysis completed successfully. Check the 'analysis_outputs' directory for results.")
    logging.info("Analysis completed successfully.")
