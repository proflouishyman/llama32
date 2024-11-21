# Date: 2024-11-01
# Purpose: Clean up text files by applying specific rules to remove unwanted markers and characters

import os
import re

# Variables
directory = '/data/lhyman6/OCR/scripts/data/second_images/'  # Specify the directory containing the text files
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
