import pandas as pd
import os
import logging

# ---------------------------- Configuration ---------------------------- #

# Path to your input CSV file
INPUT_CSV_PATH = '/data/lhyman6/OCR/scripts_newvision/llama/complete_testing_csv.csv'  # Replace with your actual CSV file path

# Directory where the .llama32 files are stored
TEXT_FILES_DIRECTORY = '/data/lhyman6/OCR/scripts/data/second_images'  # Replace with your actual directory path

# Extension of the text files
TEXT_FILE_EXTENSION = '.llama32'

# Name of the new column to be added
NEW_COLUMN_NAME = 'LLAMA32_BASE'

# Path for the output CSV file
OUTPUT_CSV_PATH = 'complete_with_llama32.csv'  # You can change this as needed

# Path for the log file
LOG_FILE_PATH = 'processing.log'  # Log file to record processing details and errors

# ---------------------------- Logging Setup ---------------------------- #

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='w',  # Overwrite the log file each time the script runs
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------- Processing ---------------------------- #

def add_file_content_to_csv(input_csv, text_dir, file_ext, new_col, output_csv):
    # Initialize counters
    total_files = 0
    successful_files = 0
    error_files = 0
    error_details = []

    # Read the CSV into a pandas DataFrame
    try:
        df = pd.read_csv(input_csv)
        logging.info(f"Successfully read the CSV file: {input_csv}")
    except FileNotFoundError:
        logging.error(f"The file {input_csv} does not exist.")
        print(f"Error: The file {input_csv} does not exist. Check the log for details.")
        return
    except pd.errors.EmptyDataError:
        logging.error("The CSV file is empty.")
        print("Error: The CSV file is empty. Check the log for details.")
        return
    except pd.errors.ParserError:
        logging.error("The CSV file is malformed.")
        print("Error: The CSV file is malformed. Check the log for details.")
        return

    # Initialize the new column with empty strings or NaN
    df[new_col] = ''

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        total_files += 1
        file_id = row['id']  # Assuming the first column is named 'id'
        file_name = f"{file_id}{file_ext}"
        file_path = os.path.join(text_dir, file_name)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                df.at[index, new_col] = content
                successful_files += 1
                logging.info(f"Added content from {file_name} to row {index}.")
            except Exception as e:
                error_files += 1
                error_message = f"Error reading {file_name}: {e}"
                error_details.append(error_message)
                logging.error(error_message)
                df.at[index, new_col] = f"Error reading file: {e}"
        else:
            error_files += 1
            error_message = f"File {file_name} does not exist. Leaving the cell empty."
            error_details.append(error_message)
            logging.warning(error_message)
            df.at[index, new_col] = ''  # Or you can put a placeholder like 'File not found'

    # Save the updated DataFrame to a new CSV
    try:
        df.to_csv(output_csv, index=False)
        logging.info(f"Successfully saved the updated CSV to {output_csv}")
    except Exception as e:
        logging.error(f"Error saving the updated CSV: {e}")
        print(f"Error: Could not save the updated CSV. Check the log for details.")
        return

    # Print and log the summary
    summary = (
        f"Processing Completed:\n"
        f"Total files processed: {total_files}\n"
        f"Successful file additions: {successful_files}\n"
        f"Files with errors: {error_files}"
    )
    print(summary)
    logging.info(summary)

    # If there were errors, provide details
    if error_files > 0:
        error_summary = "Error Details:\n" + "\n".join(error_details)
        print(error_summary)
        logging.info(error_summary)

# ---------------------------- Execution ---------------------------- #

if __name__ == "__main__":
    add_file_content_to_csv(
        input_csv=INPUT_CSV_PATH,
        text_dir=TEXT_FILES_DIRECTORY,
        file_ext=TEXT_FILE_EXTENSION,
        new_col=NEW_COLUMN_NAME,
        output_csv=OUTPUT_CSV_PATH
    )
