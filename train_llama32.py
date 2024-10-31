import os
import json
import torch
import logging
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import SFTConfig, SFTTrainer

# ----------------------------
# Logging Configuration
# ----------------------------

LOG_FILE = 'llama_train.log'

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
TRAIN_JSON = 'training/annotations/train.json'
TEST_JSON = 'training/annotations/test.json'
OUTPUT_DIR = 'my-awesome-llama'
BATCH_SIZE = 4  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
BF16 = True  # Use bfloat16 for NVIDIA L40s
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Functions
# ----------------------------

def collate_fn(batch, processor):
    texts = [item["messages"] for item in batch]
    images = []
    for item in batch:
        image_path = item["images"][0]
        try:
            with Image.open(image_path).convert("RGB") as img:
                images.append(img.copy())
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            images.append(Image.new("RGB", (224, 224)))  # Placeholder image

    # Apply chat template if necessary
    processed_texts = [processor.apply_chat_template(messages, tokenize=False) for messages in texts]

    # Tokenize texts and process images
    encoding = processor(text=processed_texts, images=images, return_tensors="pt", padding=True)

    # Prepare labels
    labels = encoding["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100  # Mask image tokens if necessary
    encoding["labels"] = labels

    # Move tensors to the appropriate device
    for key in encoding:
        encoding[key] = encoding[key].to(DEVICE)

    return encoding

# ----------------------------
# Main Training Function
# ----------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load processor and model
    logger.info("Loading processor and model...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if BF16 else torch.float32
        ).to(DEVICE)
        logger.info("Processor and model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load processor or model: {e}")
        return

    # Load datasets
    logger.info("Loading datasets...")
    try:
        data_files = {"train": TRAIN_JSON, "test": TEST_JSON}
        dataset = load_dataset('json', data_files=data_files)
        logger.info(f"Datasets loaded successfully. Training samples: {len(dataset['train'])}, Testing samples: {len(dataset['test'])}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # Define a wrapper for the collate_fn to include the processor
    def wrapped_collate_fn(batch):
        return collate_fn(batch, processor)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        bf16=BF16,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        remove_unused_columns=False,
        push_to_hub=False  # Set to True to push the model to Hugging Face Hub
    )

    # Initialize the trainer with progress indicators
    logger.info("Initializing trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=wrapped_collate_fn,
            tokenizer=processor.tokenizer,
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return

    # Start training with a progress bar
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training started successfully.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        return

    # Save the fine-tuned model
    logger.info("Saving the model...")
    try:
        trainer.save_model(OUTPUT_DIR)
        logger.info(f"Model saved successfully to '{OUTPUT_DIR}'.")
    except Exception as e:
        logger.error(f"Failed to save the model: {e}")
        return

