import os
import time
import torch
import statistics
import cProfile
import pstats
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

# Configuration
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
image_directory = "/data/lhyman6/OCR/scripts_newvision/sample"
prompt_file = "test_prompt.txt"
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
output_file = "model_responses.txt"    # File to save responses
profiling_log_file = "profiling_log.txt"  # File to save profiling log

batch_size = 4  # Adjust based on your GPU's capacity

def main():
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0  # Optional: Adjust as needed
    )

    # Load the model with 8-bit quantization
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
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

    # Initialize batch lists
    batched_images = []
    batched_input_texts = []
    batched_valid_image_files = []

    # Process images in batches
    for idx, image_file in enumerate(image_files, start=1):
        # Load the image
        image_path = os.path.join(image_directory, image_file)
        try:
            image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB
            batched_images.append(image)
            batched_valid_image_files.append(image_file)
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

        batched_input_texts.extend(input_texts)

        # Check if batch is ready to be processed
        if len(batched_images) == batch_size or idx == len(image_files):
            print(f"Processing batch {idx - len(batched_images) + 1}-{idx} out of {len(image_files)} images")

            # Start timing
            start_time = time.time()

            # Tokenize inputs
            inputs = processor(
                batched_images,
                batched_input_texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            # Generate output
            with torch.no_grad():
                generation_output = model.generate(
                    **inputs,
                    max_new_tokens=512,              # Adjust based on desired response length
                    return_dict_in_generate=True,    # Enable structured output
                    output_scores=False,             # Disable scores to save memory
                    num_beams=5,                     # Use beam search with 5 beams
                    do_sample=False,                 # Not sampling, using beam search
                )

            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            print(f"Processing time for batch: {processing_time:.2f} seconds")

            # Process outputs for each item in the batch
            for i, generated_sequence in enumerate(generation_output.sequences):
                # Decode the generated sequence into text
                decoded_output = processor.decode(generated_sequence, skip_special_tokens=True)

                # Extract and display the response
                response = extract_response(decoded_output, batched_input_texts[i])

                image_idx = idx - len(batched_images) + i + 1
                image_file_name = batched_valid_image_files[i]
                print(f"Image {image_idx}: {image_file_name}")
                print("Generated Response:")
                print(response)
                print("\n" + "="*50 + "\n")

                # Optional: Save responses to a file
                with open(output_file, "a") as f:
                    f.write(f"Image {image_idx}: {image_file_name}\n")
                    f.write(f"Processing time: {processing_time:.2f} seconds\n")
                    f.write(f"{response}\n")
                    f.write("="*50 + "\n")

            # Reset batches
            batched_images = []
            batched_input_texts = []
            batched_valid_image_files = []

    # Calculate statistics
    if processing_times:
        min_time = min(processing_times)
        max_time = max(processing_times)
        mean_time = statistics.mean(processing_times)
        median_time = statistics.median(processing_times)
        stdev_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0

        print("\nProcessing Time Statistics:")
        print(f"Total batches processed: {len(processing_times)}")
        print(f"Minimum batch time: {min_time:.2f} seconds")
        print(f"Maximum batch time: {max_time:.2f} seconds")
        print(f"Mean batch time: {mean_time:.2f} seconds")
        print(f"Median batch time: {median_time:.2f} seconds")
        print(f"Standard deviation: {stdev_time:.2f} seconds")
    else:
        print("No batches were processed.")

    print("Processing completed.")

if __name__ == "__main__":
    # Profile the main function and write the stats to a file
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    with open(profiling_log_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')  # Sort by cumulative time
        stats.print_stats()
    print(f"Profiling completed. Log saved to {profiling_log_file}")
