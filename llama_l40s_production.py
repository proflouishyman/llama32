import os
import torch
from pathlib import Path
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import logging

# ============================
# Configuration Variables
# ============================

# Logging Configuration
LOG_BASE_NAME = 'llama_l40s_log'
LOG_EXTENSION = '.log'
LOG_LEVEL = logging.DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Model and Processor Configuration
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
TORCH_DTYPE = torch.float16
DEVICE_MAP = "auto"

# File Paths
PROMPT_FILE = Path("gompers_prompt.txt")
IMAGE_DIRECTORY = Path("/data/lhyman6/OCR/scripts/data/second_images").resolve()

# File Handling
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
OUTPUT_SUFFIX = ".llama32"
LOCK_SUFFIX = ".lock"

# Processing Parameters
MAX_NEW_TOKENS = 1000

# ============================
# Helper Functions
# ============================

def get_log_file_name(base_name=LOG_BASE_NAME, extension=LOG_EXTENSION):
    """Generates a unique log file name by appending an incrementing number."""
    i = 1
    log_file_name = f"{base_name}{i}{extension}"
    
    while os.path.exists(log_file_name):
        i += 1
        log_file_name = f"{base_name}{i}{extension}"
    
    return log_file_name

# ============================
# Main Processing Function
# ============================

def process_images():
    # Setup logging
    logger = logging.getLogger('LlamaProcessor')
    logger.setLevel(LOG_LEVEL)  # Set logging level based on configuration
    
    # Prevent adding multiple handlers if the function is called multiple times
    if not logger.handlers:
        # Define log format
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Create console handler and set level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(log_format)

        # Create file handler and set level
        log_file_name = get_log_file_name()
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.info(f"Logging initialized. Log file: {log_file_name}")

    # Load the model with appropriate settings
    logger.info("Loading model...")
    try:
        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,  # Use configured dtype for efficiency
            device_map=DEVICE_MAP,
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Tie the input and output embeddings
    try:
        model.tie_weights()
        logger.info("Model weights tied successfully.")
    except Exception as e:
        logger.error(f"Failed to tie model weights: {e}")
        return

    # Set model to evaluation mode
    model.eval()

    # Load the processor
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        logger.info("Processor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        return

    # Read the prompt from a text file
    if not PROMPT_FILE.is_file():
        logger.error(f"Prompt file {PROMPT_FILE} does not exist.")
        return
    try:
        with PROMPT_FILE.open("r", encoding="utf-8") as file:
            full_prompt_text = file.read()
        logger.info(f"Prompt loaded from {PROMPT_FILE}.")
    except Exception as e:
        logger.error(f"Failed to read prompt file: {e}")
        return

    # Verify the image directory exists
    if not IMAGE_DIRECTORY.is_dir():
        logger.error(f"Image directory {IMAGE_DIRECTORY} does not exist.")
        return

    # List of image files in the directory
    image_files = sorted([f for f in IMAGE_DIRECTORY.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS])

    total_images = len(image_files)

    if not image_files:
        logger.info("No images found to process.")
        return

    logger.info(f"Found {total_images} image file(s) to process.")

    # Initialize counters
    processed_count = 0
    skipped_count = 0

    # Initialize processing index
    index = 0

    while index < total_images:
        image_path = image_files[index]
        output_file = image_path.with_suffix(image_path.suffix + OUTPUT_SUFFIX)
        lock_file = image_path.with_suffix(image_path.suffix + LOCK_SUFFIX)

        # Debug logs for paths being checked
        logger.debug(f"Checking if output file {output_file} exists.")
        logger.debug(f"Checking if lock file {lock_file} exists.")

        # Check if the output file already exists
        if output_file.exists():
            skipped_count += 1
            logger.info(f"Skipping {image_path.name}: already processed. ({index + 1} of {total_images})")
            index += 1  # Move to the next file
            continue

        # Check if a lock file exists (another instance is processing)
        if lock_file.exists():
            skipped_count += 1
            logger.info(f"Skipping {image_path.name}: currently being processed by another instance. ({index + 1} of {total_images})")
            index += 1  # Move to the next file
            continue

        try:
            # Create a lock file to indicate processing
            lock_file.touch(exist_ok=False)
            logger.info(f"Lock acquired for {image_path.name}")

            # Open the image
            with image_path.open("rb") as img_file:
                image = Image.open(img_file).convert("RGB")

            logger.info(f"Processing {image_path.name} ({index + 1} of {total_images})")

            # Prepare the messages for the model
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt_text}
                ]}
            ]

            # Apply the chat template to generate input text
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            # Process the image and text to create model inputs
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            # Generate the model's output
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

            # Decode the output
            decoded_output = processor.decode(output[0])

            # Save the decoded output to the file
            with output_file.open("w", encoding="utf-8") as out_file:
                out_file.write(decoded_output)

            processed_count += 1
            logger.info(f"Processed {image_path.name} ({processed_count} of {total_images}): saved to {output_file.name}")

            # Remove the lock file after successful processing
            lock_file.unlink()
            logger.info(f"Lock released for {image_path.name}")

            # Move to the next file
            index += 1

        except Exception as e:
            skipped_count += 1
            logger.error(f"An error occurred while processing {image_path.name}: {e} ({index + 1} of {total_images})")
            # Remove the lock file in case of an error to prevent deadlocks
            if lock_file.exists():
                lock_file.unlink()
                logger.info(f"Lock released for {image_path.name} due to error.")
            # Move to the next file after an error
            index += 1

    # Summary of processing
    logger.info("Processing Complete.")
    logger.info(f"Total Images Found: {total_images}")
    logger.info(f"Total Images Processed: {processed_count}")
    logger.info(f"Total Images Skipped: {skipped_count}")

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    process_images()
