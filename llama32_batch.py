import os
import time
import torch
import statistics
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from torch.cuda.amp import autocast

# Configuration
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
image_directory = "/data/lhyman6/OCR/scripts_newvision/sample"
prompt_file = "test_prompt.txt"
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
output_file = "model_responses.txt"  # Optional: File to save responses

# Load the model and processor once
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.tie_weights()  # Tie the input and output embeddings
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

def extract_response(full_output, prompt):
    """
    Extracts the response part from the model's output by removing the prompt.

    Parameters:
        full_output (str): The full generated text from the model.
        prompt (str): The exact prompt sent to the model.

    Returns:
        str: The extracted response.
    """
    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    else:
        # Attempt to find the prompt within the output
        prompt_index = full_output.find(prompt)
        if prompt_index != -1:
            return full_output[prompt_index + len(prompt):].strip()
    # Fallback to returning the full output
    return full_output.strip()

# Initialize list to store processing times
processing_times = []

# Optional: Initialize output file
with open(output_file, "w") as f:
    f.write("Model Responses:\n\n")

# Process images individually
for idx, image_file in enumerate(image_files, start=1):
    images = []
    valid_image_files = []

    # Load the image
    image_path = os.path.join(image_directory, image_file)
    try:
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB
        images.append(image)
        valid_image_files.append(image_file)
    except Exception as e:
        print(f"Failed to load {image_file}: {e}")
        continue  # Skip to next image if failed to load

    # Prepare messages and input texts
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt_text}
                ]
            }
        ]
    ]

    input_texts = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Ensure input_texts is a list of strings
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    print(f"Processing Image {idx}/{len(image_files)}: {image_file}")
    print(f"Input text: {input_texts[0]}")

    # Verify the number of image tokens
    image_token_count = input_texts[0].count('<|image|>')
    print(f"Input text has {image_token_count} image token(s)")

    try:
        # Start timing
        start_time = time.time()

        # Tokenize inputs
        inputs = processor(
            images,
            input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        # Generate output
        with torch.no_grad():  # Disable gradient calculations for inference
            with autocast():
                generation_output = model.generate(
                    **inputs,
                    max_new_tokens=2000,              # Adjust based on desired response length
                    return_dict_in_generate=True,    # Enable structured output
                    output_scores=False,             # Disable scores to save memory
                    output_attentions=False,         # Disable attentions to save memory
                )

        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)

        print(f"Processing time for {image_file}: {processing_time:.2f} seconds")

        # Access the generated sequence
        generated_sequence = generation_output.sequences[0]  # Tensor of shape (sequence_length,)

        # Decode the generated sequence into text
        decoded_output = processor.decode(generated_sequence, skip_special_tokens=True)

        # Extract and display the response
        response = extract_response(decoded_output, input_texts[0])

        print("Generated Response:")
        print(response)
        print("\n" + "="*50 + "\n")

        # Optional: Save responses to a file
        with open(output_file, "a") as f:
            f.write(f"Image {idx}: {image_file}\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n")
            f.write(f"{response}\n")
            f.write("="*50 + "\n")

    except Exception as e:
        print(f"An error occurred while processing {image_file}: {e}")

# Calculate statistics
if processing_times:
    min_time = min(processing_times)
    max_time = max(processing_times)
    mean_time = statistics.mean(processing_times)
    median_time = statistics.median(processing_times)
    stdev_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0

    print("\nProcessing Time Statistics:")
    print(f"Total documents processed: {len(processing_times)}")
    print(f"Minimum time: {min_time:.2f} seconds")
    print(f"Maximum time: {max_time:.2f} seconds")
    print(f"Mean time: {mean_time:.2f} seconds")
    print(f"Median time: {median_time:.2f} seconds")
    print(f"Standard deviation: {stdev_time:.2f} seconds")
else:
    print("No documents were processed.")

print("Processing completed.")
