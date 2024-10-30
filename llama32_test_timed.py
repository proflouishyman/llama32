import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import time
import statistics
import cProfile
import pstats
from io import StringIO

def process_images():
    # Model and Processor Configuration
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # Load the model with appropriate settings
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        #device_map="auto",
        device_map={"": 0},  # Load the entire model onto GPU 0
    )
    
    # Tie the input and output embeddings
    model.tie_weights()
    
    # Set model to evaluation mode
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Read the prompt from a text file
    with open("test_prompt.txt", "r") as file:
        full_prompt_text = file.read()
    
    # Directory containing images to process
    image_directory = "/data/lhyman6/OCR/scripts_newvision/sample"
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # List of image files in the directory
    image_files = [
        f for f in os.listdir(image_directory)
        if f.lower().endswith(supported_formats)
    ]
    
    # Initialize a list to store processing times for each image
    processing_times = []
    
    # Record the start time for total processing
    total_start_time = time.time()
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        
        try:
            # Start timer for the current image
            start_time = time.time()
            
            # Open the image
            image = Image.open(image_path)

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
            output = model.generate(**inputs, max_new_tokens=500)
    
            # Decode and display the output
            print(f"Image: {image_file}")
            print(processor.decode(output[0]))
            print("\n" + "="*50 + "\n")
            
            # End timer for the current image
            end_time = time.time()
            elapsed_time = end_time - start_time
            processing_times.append(elapsed_time)
            
        except Exception as e:
            print(f"An error occurred while processing {image_file}: {e}")
    
    # Record the end time for total processing
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    # Calculate and display statistics if any images were processed
    if processing_times:
        min_time = min(processing_times)
        max_time = max(processing_times)
        median_time = statistics.median(processing_times)
        mean_time = statistics.mean(processing_times)
        stdev_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0
        
        print("Processing Time Statistics:")
        print(f"Total Time: {total_elapsed_time:.2f} seconds")
        print(f"Number of Images Processed: {len(processing_times)}")
        print(f"Range: {min_time:.2f} - {max_time:.2f} seconds")
        print(f"Median: {median_time:.2f} seconds")
        print(f"Mean: {mean_time:.2f} seconds")
        print(f"Standard Deviation: {stdev_time:.2f} seconds")
    else:
        print("No images were processed successfully.")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        process_images()
    finally:
        profiler.disable()
        # Create a stream to hold the profiling results
        s = StringIO()
        sortby = 'cumulative'  # You can sort by 'time', 'cumulative', etc.
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(50)  # Print top 50 lines of the profiling report
        print("\nProfiling Results:\n")
        print(s.getvalue())
