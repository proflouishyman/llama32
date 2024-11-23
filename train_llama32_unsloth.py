# /data/lhyman6/OCR/scripts_newvision/llama/train_llama32_unsloth.py
#later file
import os
from functools import wraps
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, TrainerCallback, TrainerState, TrainerControl
from trl import SFTTrainer, SFTConfig
import logging
import psutil
from PIL import Image
from datasets import load_dataset

from huggingface_hub import login
from unsloth import FastLanguageModel
from peft import LoraConfig

# from https://blog.futuresmart.ai/fine-tune-llama-32-vision-language-model-on-custom-datasets


# Set the MASTER_PORT and MASTER_ADDR for distributed training (if using SLURM)
MASTER_PORT = os.getenv('MASTER_PORT', '29500')  # Default to '29500' if not set
MASTER_ADDR = os.getenv('MASTER_ADDR', 'localhost') 

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = MASTER_ADDR
os.environ['MASTER_PORT'] = MASTER_PORT

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

checkpoint_dir='/scratch4/lhyman6/llama_training/unsloth/'
log_name ="llama_ica100_train_log_11_21_unsloth.log"

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
    log_file_name = log_name
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
        gpu_mem_free = gpu_mem - gpu_mem_alloc
        logger.info(f"[{stage}] GPU Memory Usage: Reserved={gpu_mem:.2f} MB, Allocated={gpu_mem_alloc:.2f} MB, Free={gpu_mem_free:.2f} MB")

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
    global processor, tokenizer  # Make processor available to collate_fn
    logger.info("Loading processor and model...")
    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=hf_token)
    model = FastLanguageModel.from_pretrained(
        model_name= MODEL_ID,
        max_seq_length=2048,
        torch_dtype=torch.float16,
        load_in_4bit=True,
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

    # Delete intermediate variables to free memory
    del messages, texts, images, labels
    torch.cuda.empty_cache()


    # logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
    # logger.info(f"Labels shape: {batch['labels'].shape}")
    return batch

# ============================
#   Debugging Evaluation
# ============================
def check_eval_dataset(trainer):
    logger.info("Checking evaluation dataset...")
    eval_dataset = trainer.eval_dataset
    for i in range(3):  # Check first 3 examples
        example = eval_dataset[i]
        logger.info(f"\nExample {i}:")
        logger.info(f"Input text: {example['messages']}")
        logger.info(f"Image path: {example['images'][0]}")
        
        # Test image loading
        try:
            image = load_image(example['images'][0])
            logger.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            
        # Process example through collate_fn
        batch = collate_fn([example])
        logger.info(f"Processed shapes - Input IDs: {batch['input_ids'].shape}, Labels: {batch['labels'].shape}")


# ============================
# Trainer Initialization
# ============================

@profile_memory("Initialize Trainer")
def initialize_trainer(model, dataset):
    OUTPUT_DIR = checkpoint_dir
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,          # Batch size per GPU
        gradient_accumulation_steps=8,          # Accumulate gradients over 8 steps
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True, #not bf16
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        remove_unused_columns=False,
        dataset_kwargs={
            "skip_prepare_dataset": True        #needed for multimodal
        },
        save_total_limit=3,

        #use more cpus
        dataloader_num_workers=12,
        
    )


    # Initialize the custom callback
    memory_callback = FreeMemoryCallback(interval="step")  # Options: "epoch", "step", "both"


   


    trainer = FastLanguageModel.get_peft_model(
        model, #replace trainer in the original code per o1
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
 # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_fn,
        tokenizer=tokenizer,
        peft_config=peft_config,
        callbacks=[memory_callback],  # Add the callback here
    )


    # Add evaluation dataset check
    check_eval_dataset(trainer)

    return trainer

    


# ==================
# Memory Management
#===================

def free_gpu_memory(stage=""):
    """
    Frees GPU memory by emptying the cache and collecting garbage.

    Args:
        stage (str): Optional description of when this function is called.
    """
    import torch
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        logger.info(f"[{stage}] Freed GPU memory")


class FreeMemoryCallback(TrainerCallback):
    """
    A Hugging Face Trainer callback to free GPU memory at specified intervals.
    """

    def __init__(self, interval="epoch"):
        """
        Initializes the callback.

        Args:
            interval (str): When to free memory. Options:
                            - "step": after every training step
                            - "epoch": after every epoch
                            - "custom": implement custom logic
        """
        self.interval = interval

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.interval in ["epoch", "both"]:
            free_gpu_memory(stage="After Epoch")
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.interval in ["step", "both"]:
            free_gpu_memory(stage=f"After Step {state.global_step}")
        return control


# ============================
# QOL Functions
# ============================
def create_checkpoint_dir(directory):
    """
    Creates the checkpoint directory if it doesn't exist.

    Args:
        directory (str): Path to the checkpoint directory.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created checkpoint directory at: {directory}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory {directory}: {e}")
            raise
    else:
        logger.info(f"Checkpoint directory already exists at: {directory}")



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

    # Determine the checkpoint to resume from, if any
    checkpoint = None
    OUTPUT_DIR = checkpoint_dir

    # **Create the checkpoint directory if it doesn't exist**
    create_checkpoint_dir(OUTPUT_DIR)

    # Search for checkpoints in the output directory
    checkpoints = list(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    if checkpoints:
        # Sort checkpoints by modification time (latest first)
        checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)
        checkpoint = checkpoints[0]
        logger.info(f"Found checkpoint: {checkpoint}")
    else:
        logger.info("No checkpoints found. Starting training from scratch.")

    # Log memory before training
    log_memory_usage("Before Training")

    free_gpu_memory("Pre-Training")

    # Train with memory profiling
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)


    # Assuming 'trainer' is your SFTTrainer object from the previous step
    peft_model = trainer.model

    # Save the PEFT weights
    peft_model.save_pretrained(OUTPUT_DIR)
    # Save the tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Log memory after training
    log_memory_usage("After Training")

    # Save with memory profiling
    logger.info("Saving model and processor...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Log final memory usage
    log_memory_usage("End")

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    main()