import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Configuration
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
image_directory = "/data/lhyman6/OCR/scripts_newvision/sample"
prompt_file = "rolls_prompt.txt"
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
batch_size = 8  # Set your desired batch size here

# Load the model and processor once
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()  # Set model to evaluation mode
processor = AutoProcessor.from_pretrained(model_id)

# Read the prompt from the file
with open(prompt_file, "r") as file:
    full_prompt_text = file.read()

# Get list of image files
image_files = [
    f for f in os.listdir(image_directory)
    if f.lower().endswith(supported_formats)
]

def chunked_iterable(iterable, size):
    """Yield successive chunks from iterable of given size."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Process images in batches
for batch_num, image_batch in enumerate(chunked_iterable(image_files, batch_size), start=1):
    images = []
    valid_image_files = []
    
    # Load images and prepare inputs
    for image_file in image_batch:
        image_path = os.path.join(image_directory, image_file)
        try:
            image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB
            images.append(image)
            valid_image_files.append(image_file)
        except Exception as e:
            print(f"Failed to load {image_file}: {e}")
    
    if not images:
        continue  # Skip to next batch if no images are valid

    # Prepare messages and input texts
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": full_prompt_text}
        ]}
    ] * len(images)  # Duplicate the messages for each image in the batch

    input_texts = processor.apply_chat_template(messages, add_generation_prompt=True)

    try:
        # Tokenize inputs
        inputs = processor(
            images,
            input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,  # Pad sequences to the same length
            truncation=True  # Truncate sequences that are too long
        ).to(model.device)
        
       with torch.no_grad():  # Disable gradient calculations for inference
        generation_output = model.generate(
            **inputs,
            max_new_tokens=200,                 # Adjust based on desired response length
            return_dict_in_generate=True,       # Enable structured output
            output_scores=False,                # Disable scores to save memory
            output_attentions=False,            # Disable attentions to save memory
        )
        # Decode and display outputs
        decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
        
        for img_file, output in zip(valid_image_files, decoded_outputs):
            print(f"Image: {img_file}")
            print(output)
            print("\n" + "="*50 + "\n")
    
    except Exception as e:
        print(f"An error occurred while processing batch {batch_num}: {e}")

print("Batch processing completed.")
