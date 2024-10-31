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
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
BF16 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_TOKEN = "<|image|>"

def load_and_verify_image(image_path):
    """Load and verify an image, returning a placeholder if there's an error."""
    try:
        with Image.open(image_path).convert("RGB") as img:
            return img.copy()
    except Exception as e:
        logger.warning(f"Error loading image {image_path}: {e}")
        return Image.new("RGB", (224, 224))
    
def process_text_with_image_token(messages, processor):
    """Process text and add image token in the correct position."""
    try:
        # Extract the user query and assistant response
        user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
        assistant_message = next(msg["content"] for msg in messages if msg["role"] == "assistant")
        
        # Clean and standardize the text
        user_message = user_message.strip()
        assistant_message = assistant_message.strip()
        
        # Create input and target text
        input_text = f"{IMAGE_TOKEN} User: {user_message}"
        target_text = f"Assistant: {assistant_message}"
        
        return input_text, target_text
    except Exception as e:
        logger.error(f"Failed to process text: {e}")
        raise

def verify_token_counts(encoding, image_token_id, num_images, processor, processed_texts):
    """Verify that the number of image tokens matches the number of images."""
    total_image_tokens = (encoding["input_ids"] == image_token_id).sum().item()
    
    logger.info(f"Total image tokens found: {total_image_tokens}, Expected: {num_images}")
    
    if total_image_tokens != num_images:
        logger.error(f"Image token mismatch: found {total_image_tokens} tokens for {num_images} images")
        for idx, text in enumerate(processed_texts):
            tokens = processor.tokenizer.tokenize(text)
            logger.debug(f"Sample {idx} tokens: {tokens}")
        raise ValueError(f"Token mismatch: {total_image_tokens} tokens for {num_images} images")

def collate_fn(batch, processor):
    """Collate function for the DataLoader."""
    texts = [item["messages"] for item in batch]
    images = [load_and_verify_image(item["images"][0]) for item in batch]
    
    # Process texts and get target texts
    target_texts = []
    for messages in texts:
        assistant_message = next(msg["content"] for msg in messages if msg["role"] == "assistant")
        target_texts.append(assistant_message)
    
    # Debug logging
    for idx, target in enumerate(target_texts):
        logger.debug(f"Sample {idx} target: {target[:100]}...")
    
    try:
        # Encode images
        model_inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        # Encode target sequences
        with processor.tokenizer.as_target_tokenizer():
            labels = processor.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=processor.tokenizer.model_max_length,
            )["input_ids"]
            
        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Add labels to the encoding dictionary
        model_inputs["labels"] = labels

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

    # Keep tensors on CPU for proper memory pinning
    return {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_inputs.items()}


def setup_tokenizer_and_model(processor, model):
    """Set up tokenizer with image token and resize model embeddings."""
    if IMAGE_TOKEN not in processor.tokenizer.get_vocab():
        logger.info(f"Adding {IMAGE_TOKEN} to tokenizer vocabulary")
        special_tokens_dict = {'additional_special_tokens': [IMAGE_TOKEN]}
        num_added_tokens = processor.tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_tokens} special tokens to tokenizer")
        
        model.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"Resized model embeddings to {len(processor.tokenizer)}")

def setup_training_args():
    """Set up training arguments."""
    return SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        bf16=BF16,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        label_names=["labels"],
        max_length=512,  # Add explicit max length
        padding="max_length",  # Force padding to max_length
        truncation=True,  # Enable truncation
    )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("Loading processor and model...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if BF16 else torch.float32,
            device_map="auto"  # Better memory management
        )
        
        setup_tokenizer_and_model(processor, model)
        logger.info("Processor and model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load processor or model: {e}")
        raise

    logger.info("Loading datasets...")
    try:
        data_files = {"train": TRAIN_JSON, "test": TEST_JSON}
        dataset = load_dataset('json', data_files=data_files)
        logger.info(f"Datasets loaded. Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

    def wrapped_collate_fn(batch):
        return collate_fn(batch, processor)

    logger.info("Initializing trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=setup_training_args(),
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=wrapped_collate_fn,
            tokenizer=processor.tokenizer,
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise

    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

    logger.info("Saving model...")
    try:
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        logger.info(f"Model and processor saved to '{OUTPUT_DIR}'")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise