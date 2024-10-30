import os
import sys
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import gc
import logging

# Set up logging
logging.basicConfig(filename='batch_size_test.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load the model once
logging.info("Loading model...")
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Using BF16 precision
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()
processor = AutoProcessor.from_pretrained(model_id)
logging.info("Model loaded.")

# Paths to your sample images (replace with your actual image paths)
image_paths = [
    "/data/lhyman6/OCR/scripts_newvision/sample/51185-247-0101.jpg",
    "/data/lhyman6/OCR/scripts_newvision/sample/mss511850144-29.jpg",
    "/data/lhyman6/OCR/scripts_newvision/sample/RDApp-206939Doresey001.jpg",
    "/data/lhyman6/OCR/scripts_newvision/sample/RDApp-206939Doresey003.jpg",
    "/data/lhyman6/OCR/scripts_newvision/sample/RDApp-206939Doresey004.jpg",
]

# Ensure all images exist
for path in image_paths:
    if not os.path.isfile(path):
        logging.error(f"Image file not found: {path}")
        raise FileNotFoundError(f"Image file not found: {path}")

# Load images once
images = [Image.open(path) for path in image_paths]
num_images = len(images)

# Function to test a given batch size
def test_batch_size(batch_size, images):
    try:
        # Prepare inputs
        num_repeats = (batch_size + num_images - 1) // num_images  # Ceiling division
        images_batch = (images * num_repeats)[:batch_size]

        # Prepare messages for each item in the batch
        messages_list = [
            [
                {"role": "user", "content": "<image> If I had to write a haiku for this one, it would be: "}
            ]
            for _ in range(batch_size)
        ]

        # Generate input_texts for each message
        input_texts = [
            processor.apply_chat_template(msg, add_generation_prompt=True)
            for msg in messages_list
        ]

        inputs = processor(
            images=images_batch,
            text=input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        # Run inference
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)

        # Clean up
        del inputs
        del output
        torch.cuda.empty_cache()
        gc.collect()

        return True  # Successful
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            return False  # Out of memory
        else:
            raise e  # Other errors

# Initialize variables for binary search
logging.info("Starting batch size testing with binary search...")

max_batch_size = None
low = 1
high = 12  # Adjust as needed based on your GPU's capacity

while low <= high:
    mid = (low + high) // 2
    logging.info(f"Testing batch size: {mid}")
    success = test_batch_size(mid, images)
    if success:
        logging.info(f"Batch size {mid} succeeded.")
        max_batch_size = mid
        low = mid + 1  # Try higher batch sizes
    else:
        logging.info(f"Batch size {mid} failed due to out of memory.")
        high = mid - 1  # Try lower batch sizes

if max_batch_size is not None:
    logging.info(f"Optimal batch size is: {max_batch_size}")
    print(f"Optimal batch size is: {max_batch_size}")
else:
    logging.info("Failed to find a suitable batch size.")
    print("Failed to find a suitable batch size.")
