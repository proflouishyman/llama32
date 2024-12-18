import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import SFTTrainer, SFTConfig
import logging



# Configuration
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
TRAIN_JSON = 'training/annotations/train.json'
TEST_JSON = 'training/annotations/test.json'
OUTPUT_DIR = 'ica100_llama'
PROMPT_FILE = os.getenv('PROMPT_FILE', 'gompers_prompt.txt')


# ============================
# Logging
# ============================
# Set up logging configuration
LOG_BASE_NAME = 'llama_l40s_train_log'
LOG_EXTENSION = '.log'
LOG_LEVEL = logging.DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

def get_log_file_name(base_name=LOG_BASE_NAME, extension=LOG_EXTENSION):
    """Generates a unique log file name by appending an incrementing number."""
    i = 1
    log_file_name = f"{base_name}{i}{extension}"
    
    while os.path.exists(log_file_name):
        i += 1
        log_file_name = f"{base_name}{i}{extension}"
    
    return log_file_name

# Create logger
logger = logging.getLogger('app_logger')  # Use a generic logger name
logger.setLevel(LOG_LEVEL)

# Prevent adding multiple handlers
if not logger.handlers:
    # Define log format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(log_format)

    # Create file handler and set level
    log_file_name = get_log_file_name()  # Use the unique log file name function
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file_name}")



# Read prompt from file
try:
    with open(PROMPT_FILE, 'r') as f:
        USER_PROMPT = f.read().strip()
    logger.info(f"Loaded prompt from {PROMPT_FILE}")
except FileNotFoundError:
    logger.warning(f"Prompt file {PROMPT_FILE} not found, using default prompt")
    USER_PROMPT = "EXTRACT TEXT EXACTLY AND COMPLETELY AS WRITTEN. Preserve indents and whitespaces."

def load_image(image_path):
    """Load a single image."""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        # Return a small black image as fallback
        return Image.new('RGB', (224, 224))

def collate_fn(examples):

    # print("Example keys:", examples[0].keys()) #for debug

    #  # Print the actual messages
    # print("First example messages:", examples[0]['messages'])

    

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
    
    #testing 
    # for text in texts:
    #     print(text)

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

def main():
    # Load processor and model
    global processor  # Make processor available to collate_fn
    logger.info("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, #switched from bfloat16 to float16
        #device_map="auto"  #needs to be commented out for accelerate and deepspeed
    )

    # Enable gradient checkpointing if possible
    model.gradient_checkpointing_enable()


    # Log the prompt being used
    logger.info(f"Using prompt: {USER_PROMPT}")

    # Load dataset
    logger.info("Loading datasets...")
    dataset = load_dataset('json', data_files={
        "train": TRAIN_JSON,
        "test": TEST_JSON
    })
    logger.info(f"Loaded {len(dataset['train'])} training and {len(dataset['test'])} test examples")
    # Define constants
    OUTPUT_DIR = "./output"

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,          # Align with train_micro_batch_size_per_gpu
        gradient_accumulation_steps=2,          # Align with JSON
        gradient_checkpointing=True,
        bf16=False,                             # Disable BF16
        fp16=True,                              # Enable FP16
        remove_unused_columns=False,
        dataset_kwargs={
            "skip_prepare_dataset": True
        },  # Crucial for vision training
        deepspeed="deepspeed_config.json",      # Specify DeepSpeed config file
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        
        # Dynamic learning rate and scheduler settings
        learning_rate=3e-5,                     # Initial learning rate
        lr_scheduler_type="linear",             # Scheduler type (e.g., linear, cosine)
        num_train_epochs=3,                     # Total number of training epochs
        warmup_steps=1000,                      # Number of warmup steps
        weight_decay=3e-7,                      # Weight decay rate
    )




    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_fn,
        #tokenizer=processor.tokenizer, #this fixes teh preprocessing error? maybe?
    )
    
    
    
    
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()