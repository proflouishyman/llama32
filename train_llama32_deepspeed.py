import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import SFTTrainer, SFTConfig
import logging
import psutil
from PIL import Image
from datasets import load_dataset
from functools import wraps
from huggingface_hub import login

# ============================
# Authenticate with HuggingFace
# ============================

hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token:
    login(token=hf_token)
else:
    raise EnvironmentError("HUGGINGFACE_TOKEN not set. Please set it to access gated repositories.")

# ============================
# Configure Logging
# ============================

logger = logging.getLogger('app_logger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler
    log_file_name = "llama_l40s_train_log.log"
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file_name}")

# ============================
# Load Prompt
# ============================

PROMPT_FILE = os.getenv('PROMPT_FILE', 'gompers_prompt.txt')

try:
    with open(PROMPT_FILE, 'r') as f:
        USER_PROMPT = f.read().strip()
    logger.info(f"Loaded prompt from {PROMPT_FILE}")
except FileNotFoundError:
    logger.warning(f"Prompt file {PROMPT_FILE} not found, using default prompt")
    USER_PROMPT = "EXTRACT TEXT EXACTLY AND COMPLETELY AS WRITTEN. Preserve indents and whitespaces."

# ============================
# Memory Profiling Decorators
# ============================

def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 ** 2)  # Resident Set Size in MB
    vms_mb = mem_info.vms / (1024 ** 2)  # Virtual Memory Size in MB

    logger.info(f"[{stage}] CPU Memory Usage: RSS={rss_mb:.2f} MB, VMS={vms_mb:.2f} MB")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_reserved() / (1024 ** 2)  # in MB
        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"[{stage}] GPU Memory Usage: Reserved={gpu_mem:.2f} MB, Allocated={gpu_mem_alloc:.2f} MB")

def profile_memory(stage_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"--- Memory Profiling Start: {stage_name} ---")
            log_memory_usage(f"Before {stage_name}")
            result = func(*args, **kwargs)
            log_memory_usage(f"After {stage_name}")
            logger.info(f"--- Memory Profiling End: {stage_name} ---\n")
            return result
        return wrapper
    return decorator

# ============================
# Data Loading Functions
# ============================

def load_image(image_path):
    """Load a single image."""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        # Return a small black image as fallback
        return Image.new('RGB', (224, 224))

@profile_memory("Load Processor and Model")
def load_processor_and_model():
    global processor  # Make processor available to collate_fn
    logger.info("Loading processor and model...")
    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        token=hf_token
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

@profile_memory("Load Dataset")
def load_datasets():
    logger.info("Loading datasets...")
    TRAIN_JSON = 'training/annotations/train.json'
    TEST_JSON = 'training/annotations/test.json'
    dataset = load_dataset('json', data_files={
        "train": TRAIN_JSON,
        "test": TEST_JSON
    })
    logger.info(f"Loaded {len(dataset['train'])} training and {len(dataset['test'])} test examples")
    return dataset

def collate_fn(examples):
    # Create messages with our extraction prompt
    messages = [
        [
            {"role": "user", "content": f"{USER_PROMPT}{processor.image_token}"},
            {"role": "assistant", "content": example["messages"][1]["content"]}
        ] 
        for example in examples
    ]

    # Apply chat template to formatted messages
    texts = [processor.apply_chat_template(msg, tokenize=False) for msg in messages]

    # Load images as PIL Image objects
    images = [load_image(example["images"][0]) for example in examples]

    # Process both images and text together
    batch = processor(
        text=texts,
        images=images,  # Now passing PIL Image objects
        return_tensors="pt",
        padding=True
    )

    # Create labels from input_ids
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image token in labels
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    
    batch["labels"] = labels
    return batch

# ============================
# Trainer Initialization
# ============================

@profile_memory("Initialize Trainer")
def initialize_trainer(model, dataset):
    OUTPUT_DIR = 'l40s'
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,          # Batch size per GPU
        gradient_accumulation_steps=4,          # Accumulate gradients over 8 steps
        gradient_checkpointing=True,
        fp16=True,                              # Enable FP16
        remove_unused_columns=False,
        dataset_kwargs={
            "skip_prepare_dataset": True        #needed for multimodal
        },
        deepspeed="deepspeed_config.json",      # Specify DeepSpeed config file
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,

        # Learning rate and scheduler settings
        learning_rate=3e-5,                     # Initial learning rate
        lr_scheduler_type="linear",
        num_train_epochs=3,
        warmup_steps=1000,
        weight_decay=3e-7,
        max_seq_length=1024,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_fn,
    )
    return trainer

# ============================
# Main Training Function
# ============================

@profile_memory("Main Function")
def main():
    # Log initial memory usage
    log_memory_usage("Start")

    # Load processor and model with memory profiling
    model = load_processor_and_model()

    # Load datasets with memory profiling
    dataset = load_datasets()

    # Initialize trainer with memory profiling
    trainer = initialize_trainer(model, dataset)

    # Log memory before training
    log_memory_usage("Before Training")

    # Train with memory profiling
    logger.info("Starting training...")
    trainer.train()

    # Log memory after training
    log_memory_usage("After Training")

    # Save with memory profiling
    logger.info("Saving model and processor...")
    trainer.save_model('l40s')
    processor.save_pretrained('l40s')

    # Log final memory usage
    log_memory_usage("End")

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    main()
