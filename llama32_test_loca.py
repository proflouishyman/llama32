import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

#running llavaenv

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load the model and processor only once
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Read the entire content from rolls_prompt.txt
with open("rolls_prompt.txt", "r") as file:
    full_prompt_text = file.read()


# Specify the directory containing your images
image_directory = "/data/lhyman6/OCR/scripts_newvision/sample"

# Get a list of image files in the directory
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = [
    f for f in os.listdir(image_directory)
    if f.lower().endswith(supported_formats)
]

# Process each image in the directory
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    try:
        # Open the image file
        image = Image.open(image_path)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": full_prompt_text}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        # Generate the output
        output = model.generate(**inputs, max_new_tokens=200)

        # Decode and print the output
        print(f"Image: {image_file}")
        print(processor.decode(output[0]))
        print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"An error occurred while processing {image_file}: {e}")
