# Date: 2024-11-01
# Purpose: This script selects 20 random .llama32 files from the source directory,
# clears the destination directory, and copies the selected files into it.

import os
import shutil
import random

# Variables for source and destination directories
SOURCE_DIR = "/data/lhyman6/OCR/scripts/data/second_images"
DESTINATION_DIR = "/data/lhyman6/OCR/scripts_newvision/sample"
FILE_EXTENSION = ".llama32"
NUM_FILES = 20

# Step 1: Clear the destination directory
if os.path.exists(DESTINATION_DIR):
    shutil.rmtree(DESTINATION_DIR)
os.makedirs(DESTINATION_DIR)

# Step 2: Get list of .llama32 files in the source directory
llama_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(FILE_EXTENSION)]

# Step 3: Select 20 random files (or fewer if less than 20 available)
selected_files = random.sample(llama_files, min(NUM_FILES, len(llama_files)))

# Step 4: Copy each selected file to the destination directory
for file_name in selected_files:
    src_file_path = os.path.join(SOURCE_DIR, file_name)
    dest_file_path = os.path.join(DESTINATION_DIR, file_name)
    shutil.copy(src_file_path, dest_file_path)

print(f"Copied {len(selected_files)} .llama32 files to {DESTINATION_DIR}")


import os
import re

# Variables
directory = '/data/lhyman6/OCR/scripts_newvision/sample'  # Specify the directory containing the text files
file_extension = '.llama32'  # Specify the file extension to process

# Function to clean text
def clean_text(text):
    cleaned_text = re.sub(r'<\|begin_of_text\|>', '', text)  # Remove <|begin_of_text|>
    cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)  # Remove <|start_header_id|> to <|end_header_id|>
    cleaned_text = re.sub(r'<\|image\|>.*?<\|eot_id\|>', '', cleaned_text, flags=re.DOTALL)  # Remove <|image|> to <|eot_id|>
    cleaned_text = re.sub(r'<\|eot_id\|>', '', cleaned_text)  # Remove <|eot_id|>
    cleaned_text = re.sub(r'\*\*', '', cleaned_text)  # Remove double asterisks
    cleaned_text = re.sub(r'\[\w+\]', '', cleaned_text)  # Remove single words inside brackets, like [Postscript]
    return cleaned_text.strip()

# Process files in the specified directory
for filename in os.listdir(directory):
    if filename.endswith(file_extension):
        file_path = os.path.join(directory, filename)
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Write the cleaned text to a new file
        cleaned_file_path = os.path.join(directory, f"{filename}.clean")
        with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write(cleaned_text)

print("Text files cleaned and saved with '.clean' suffix.")
