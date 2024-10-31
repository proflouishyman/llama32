import os
import json
import pandas as pd
import random
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

PROMPT_FILE = 'gompers_prompt.txt'  # Use the same prompt as in the production code
CSV_FILE = '/data/lhyman6/OCR/scripts_newvision/llama/complete_testing_csv.csv'
IMAGE_DIR = "/data/lhyman6/OCR/scripts/data/second_images"  # Directory containing images
OUTPUT_DIR = 'training/annotations'
ASSISTANT_COL = 'transcription'  # Ground truth transcriptions column
IMAGE_COL = 'id'  # Filename of images
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# ----------------------------
# Functions
# ----------------------------

def load_prompt(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_csv(csv_file, assistant_col, image_col):
    df = pd.read_csv(csv_file)
    if assistant_col not in df.columns or image_col not in df.columns:
        raise ValueError("CSV must contain specified columns.")
    df = df.dropna(subset=[assistant_col, image_col])
    return df

def generate_annotations(df, prompt, image_dir, assistant_col, image_col):
    annotations = []
    success_count = 0
    error_count = 0
    skipped_files = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Annotations"):
        image_filename = row[image_col].strip() + ".jpg"  # Append .jpg extension
        assistant_response = row[assistant_col].strip()
        image_path = os.path.abspath(os.path.join(image_dir, image_filename))
        
        if not os.path.isfile(image_path):
            print(f"Warning: {image_path} does not exist. Skipping.")
            error_count += 1
            skipped_files.append(image_filename)
            continue
        
        annotations.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_response}
            ],
            "images": [image_path]
        })
        success_count += 1

    return annotations, success_count, error_count, skipped_files

def split_annotations(annotations, train_ratio, seed):
    random.seed(seed)
    random.shuffle(annotations)
    split_idx = int(len(annotations) * train_ratio)
    return annotations[:split_idx], annotations[split_idx:]

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} entries to {filepath}")

# ----------------------------
# Main Execution
# ----------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading prompt...")
    prompt = load_prompt(PROMPT_FILE)
    
    print("Loading CSV data...")
    df = load_csv(CSV_FILE, ASSISTANT_COL, IMAGE_COL)
    
    print("Generating annotations...")
    annotations, success_count, error_count, skipped_files = generate_annotations(
        df, prompt, IMAGE_DIR, ASSISTANT_COL, IMAGE_COL
    )
    
    if not annotations:
        print("No valid annotations found.")
        print(f"Total Annotations Processed: {len(df)}")
        print(f"Successful Annotations: {success_count}")
        print(f"Failed Annotations: {error_count}")
        print("No JSON files were created.")
        return
    
    print("Splitting annotations into train and test sets...")
    train_annotations, test_annotations = split_annotations(annotations, TRAIN_SPLIT_RATIO, RANDOM_SEED)
    
    print("Saving training annotations...")
    save_json(train_annotations, os.path.join(OUTPUT_DIR, 'train.json'))
    
    print("Saving testing annotations...")
    save_json(test_annotations, os.path.join(OUTPUT_DIR, 'test.json'))
    
    print("\nAnnotation creation completed successfully.")
    print(f"Total Annotations Processed: {len(df)}")
    print(f"Successful Annotations: {success_count}")
    print(f"Failed Annotations: {error_count}")
    
    if skipped_files:
        print("\nList of Skipped Files:")
        for file in skipped_files:
            print(f"- {file}")

if __name__ == "__main__":
    main()
