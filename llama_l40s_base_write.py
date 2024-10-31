import os
import torch
from pathlib import Path
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
# import time
# import statistics
import logging

def process_images():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Model and Processor Configuration
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # Load the model with appropriate settings
    logging.info("Loading model...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto",
    )
    
    # Tie the input and output embeddings
    model.tie_weights()
    
    # Set model to evaluation mode
    model.eval()
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Read the prompt from a text file
    prompt_path = Path("test_prompt.txt")
    if not prompt_path.is_file():
        logging.error(f"Prompt file {prompt_path} does not exist.")
        return
    with prompt_path.open("r", encoding="utf-8") as file:
        full_prompt_text = file.read()
    
    # Directory containing images to process
    image_directory = Path("/data/lhyman6/OCR/scripts_newvision/sample")
    if not image_directory.is_dir():
        logging.error(f"Image directory {image_directory} does not exist.")
        return
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # List of image files in the directory
    image_files = [f for f in image_directory.iterdir() if f.suffix.lower() in supported_formats]
    
    if not image_files:
        logging.info("No images found to process.")
        return
    
    # Initialize a list to store processing times for each image
    # processing_times = []
    
    # Record the start time for total processing
    # total_start_time = time.time()
    
    # Process each image
    for image_path in image_files:
        output_file = image_path.with_suffix(image_path.suffix + ".llama32")
        
        # Check if the output file already exists
        if output_file.exists():
            logging.info(f"Skipping {image_path.name}: already processed.")
            continue
        
        try:
            # Start timer for the current image
            # start_time = time.time()
            
            # Open the image
            with image_path.open("rb") as img_file:
                image = Image.open(img_file).convert("RGB")
    
            # Prepare the messages for the model
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt_text}
                ]}
            ]
            
            # Apply the chat template to generate input text
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process the image and text to create model inputs
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
    
            # Generate the model's output
            output = model.generate(**inputs, max_new_tokens=1000)
            
            # Decode the output
            decoded_output = processor.decode(output[0])
    
            # Save the decoded output to the file
            with output_file.open("w", encoding="utf-8") as out_file:
                out_file.write(decoded_output)
            
            logging.info(f"Processed and saved output for {image_path.name} to {output_file.name}")
            
            # End timer for the current image
            # elapsed_time = time.time() - start_time
            # processing_times.append(elapsed_time)
            
        except Exception as e:
            logging.error(f"An error occurred while processing {image_path.name}: {e}")
    
    # Record the end time for total processing
    # total_elapsed_time = time.time() - total_start_time
    
    # Calculate and display statistics if any images were processed
    # if processing_times:
    #     min_time = min(processing_times)
    #     max_time = max(processing_times)
    #     median_time = statistics.median(processing_times)
    #     mean_time = statistics.mean(processing_times)
    #     stdev_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0
        
    #     logging.info("\nProcessing Time Statistics:")
    #     logging.info(f"Total Time: {total_elapsed_time:.2f} seconds")
    #     logging.info(f"Number of Images Processed: {len(processing_times)}")
    #     logging.info(f"Range: {min_time:.2f} - {max_time:.2f} seconds")
    #     logging.info(f"Median: {median_time:.2f} seconds")
    #     logging.info(f"Mean: {mean_time:.2f} seconds")
    #     logging.info(f"Standard Deviation: {stdev_time:.2f} seconds")
    # else:
    #     logging.info("No images were processed successfully.")

if __name__ == "__main__":
    process_images()
