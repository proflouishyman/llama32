import os 
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Configuration
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
image_directory = "/data/lhyman6/OCR/scripts_newvision/sample"
prompt_file = "test_prompt.txt"
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
batch_size = 8  # Set your desired batch size here
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

def chunked_iterable(iterable, size):
    """Yield successive chunks from iterable of given size."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

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

# Optional: Initialize output file
with open(output_file, "w") as f:
    f.write("Model Responses:\n\n")

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
        print(f"No valid images in batch {batch_num}. Skipping...")
        continue  # Skip to next batch if no images are valid

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
        ] for _ in range(len(images))
    ]

    input_texts = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Ensure input_texts is a list of strings
    if isinstance(input_texts, str):
        input_texts = [input_texts] * len(images)

    print(f"Images length: {len(images)}")
    print(f"Input texts length: {len(input_texts)}")
    print(f"Sample input text: {input_texts[0]}")

    # Verify the number of image tokens
    for i, text in enumerate(input_texts):
        image_token_count = text.count('<|image|>')
        print(f"Text {i} has {image_token_count} image token(s)")
    print(f"Total images provided: {len(images)}")

    try:
        # Tokenize inputs
        inputs = processor(
            images,
            input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        print("Inputs created successfully")
        print(f"Input keys: {inputs.keys()}")
        print(f"Input shapes: {[(k, v.shape) for k,v in inputs.items()]}")

        with torch.no_grad():  # Disable gradient calculations for inference
            generation_output = model.generate(
                **inputs,
                max_new_tokens=200,              # Adjust based on desired response length
                return_dict_in_generate=True,     # Enable structured output
                output_scores=False,              # Disable scores to save memory
                output_attentions=False,          # Disable attentions to save memory
            )
        
        # Access the generated sequences from the ModelOutput
        generated_sequences = generation_output.sequences  # Tensor of shape (batch_size, sequence_length)
        
        # Decode the generated sequences into text
        decoded_outputs = processor.batch_decode(generated_sequences, skip_special_tokens=True)
        
        # Extract and display the responses
        for idx, decoded_output in enumerate(decoded_outputs):
            # Ensure prompt corresponds to the current image
            current_prompt = input_texts[idx]
            response = extract_response(decoded_output, current_prompt)
            image_file = valid_image_files[idx]
            print(f"Batch {batch_num}, Image {idx + 1}: {image_file}")
            print("Generated Response:")
            print(response)
            print("\n" + "="*50 + "\n")
            
            # Optional: Save responses to a file
            with open(output_file, "a") as f:
                f.write(f"Batch {batch_num}, Image {idx + 1}: {image_file}\n")
                f.write(f"{response}\n")
                f.write("="*50 + "\n")
    
    except Exception as e:
        print(f"An error occurred while processing batch {batch_num}: {e}")

print("Batch processing completed.")
