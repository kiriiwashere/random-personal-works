import os
import argparse
from PIL import Image, UnidentifiedImageError
import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import threading
# run commands
# image to folder or image
# python cap.py /path/to/your/image.jpg
# python cap.py /path/to/your/folder/
# cuda device
# python caption_images.py /path/to/your/image.jpg --device cpu
# python caption_images.py /path/to/your/image.jpg --device cuda
# torch data
# python caption_images.py /path/to/your/image.jpg --dtype float16
# python caption_images.py /path/to/your/image.jpg --dtype float32

# requirements
# pip install torch torchvision torchaudio transformers Pillow accelerate

# --- Configuration ---
BASE_MODEL_ID = "Andres77872/SmolVLM-500M-anime-caption-v0.2"
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

# --- Model Components (will be initialized in main) ---
PROCESSOR = None
MODEL = None

# --- Custom Stopping Criteria (from user's original code) ---
class StopOnTokens(StoppingCriteria):
    """
    Custom stopping criteria that stops generation when a specific sequence of tokens (string)
    is found in the generated text, primarily checking the tail end of the generation.
    """
    def __init__(self, tokenizer, stop_sequence_str: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_sequence_str = stop_sequence_str

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: tensor of shape (batch_size, sequence_length)
        # Decode the currently generated tokens for the first item in the batch.
        # The model in this script processes one image at a time, so batch_size is 1.
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # This logic, from the original script, checks if the stop_sequence_str
        # appears near the end of the generated_text.
        # max_keep defines how many characters from the end of generated_text to check.
        max_keep = len(self.stop_sequence_str) + 10 
        
        if len(generated_text) > max_keep:
            text_to_check = generated_text[-max_keep:]
        else:
            text_to_check = generated_text
            
        return self.stop_sequence_str in text_to_check

# --- Input Preparation Function (adapted from user's original code) ---
def prepare_model_inputs(image: Image.Image, processor_instance, model_instance):
    """
    Prepares the inputs for the Idefics3 model.
    The question prompt "describe the image" is fixed as per the model's design.
    """
    question = "describe the image" # This prompt is crucial for this specific captioning model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Ensure image size is compatible with the processor's expected input
    # This logic is based on the original script's handling of image size.
    max_image_edge = processor_instance.image_processor.max_image_size["longest_edge"]
    current_size_config = processor_instance.image_processor.size.copy()
    if "longest_edge" in current_size_config and current_size_config["longest_edge"] > max_image_edge:
        current_size_config["longest_edge"] = max_image_edge
    
    # Apply chat template and process inputs
    prompt_str = processor_instance.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor_instance(
        text=[prompt_str], 
        images=[[image]],  # Note: images is a list of lists of images
        return_tensors='pt', 
        padding=True, 
        size=current_size_config
    )
    
    # Move inputs to the model's device
    inputs = {k: v.to(model_instance.device) for k, v in inputs.items()}
    return inputs

# --- Image Processing and Caption Generation ---
def generate_caption_for_image(image_path: str, processor_instance, model_instance):
    """
    Generates a caption for a single image file.
    Returns the caption string, or None if an error occurs.
    """
    print(f"Processing image: {image_path}...")
    try:
        # Open and convert image to RGB
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"  Error: Image file not found at {image_path}")
        return None
    except UnidentifiedImageError:
        print(f"  Error: Cannot identify image file {image_path}. It might be corrupted or not a valid image format.")
        return None
    except Exception as e:
        print(f"  Error opening image {image_path}: {e}")
        return None

    # Prepare inputs for the model
    model_inputs = prepare_model_inputs(img, processor_instance, model_instance)
    
    # Define the stop sequence string (e.g., a specific tag indicating end of caption)
    stop_sequence_str = "</RATING>" 
    
    # Initialize the streamer for iterative text generation
    streamer = TextIteratorStreamer(
        processor_instance.tokenizer,
        skip_prompt=True,      # Do not include the input prompt in the output
        skip_special_tokens=True # Do not include special tokens (like <eos>) in the output
    )
    
    # Setup custom stopping criteria
    # This uses the StopOnTokens class defined earlier
    custom_stopping_criteria = StoppingCriteriaList([
        StopOnTokens(processor_instance.tokenizer, stop_sequence_str)
    ])

    # Define generation parameters
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        do_sample=False,        # Use greedy decoding for deterministic output
        max_new_tokens=512,     # Maximum number of tokens to generate for the caption
        pad_token_id=processor_instance.tokenizer.pad_token_id, # Pad token for batching (though batch size is 1)
        stopping_criteria=custom_stopping_criteria, # Custom criteria to stop generation
    )

    # Run model generation in a separate thread to allow streaming
    # This is important because model.generate() is blocking with a streamer
    generation_thread = threading.Thread(target=model_instance.generate, kwargs=generation_kwargs)
    generation_thread.start()

    # Collect caption parts from the streamer
    caption_parts = []
    print(f"  Caption for {os.path.basename(image_path)}: ", end="")
    for new_text_chunk in streamer:
        print(new_text_chunk, end="", flush=True) # Print chunks as they arrive
        caption_parts.append(new_text_chunk)
    print() # Newline after the full caption is printed

    generation_thread.join() # Wait for the generation thread to complete
    
    full_caption = "".join(caption_parts).strip()
    
    # Optionally, remove the stop_sequence string if it's part of the output
    if full_caption.endswith(stop_sequence_str):
        full_caption = full_caption[:-len(stop_sequence_str)].strip()
        
    return full_caption

# --- File Handling ---
def save_caption_to_file(caption: str, image_file_path: str):
    """
    Saves the generated caption to a .txt file.
    The .txt file will have the same base name as the image and be saved in the same directory.
    """
    if caption is None or not caption.strip():
        print(f"  Skipping save for {image_file_path} due to empty or no caption.")
        return
        
    # Create the output text file path (e.g., image.jpg -> image.txt)
    base_name, _ = os.path.splitext(image_file_path)
    output_txt_path = base_name + ".txt"
    
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"  Caption successfully saved to: {output_txt_path}")
    except Exception as e:
        print(f"  Error saving caption to {output_txt_path}: {e}")

# --- Main Execution Logic ---
def main():
    global PROCESSOR, MODEL # Allow assignment to global variables

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate captions for images using the SmolVLM model.")
    parser.add_argument("input_path", type=str, help="Path to a single image file or a folder containing images.")
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to run the model on (e.g., 'cuda', 'cpu', 'auto'). Default is 'auto'."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the model ('bfloat16', 'float16', 'float32'). Default is 'bfloat16'."
    )
    args = parser.parse_args()

    # --- Determine Torch Dtype ---
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    print(f"Using torch dtype: {torch_dtype}")


    # --- Load Model and Processor (once) ---
    print(f"Loading model: {BASE_MODEL_ID}...")
    try:
        PROCESSOR = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        MODEL = Idefics3ForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            device_map=args.device,  # Use device specified by user or 'auto'
            torch_dtype=torch_dtype  # Use specified torch dtype
        )
        MODEL.eval() # Set model to evaluation mode (important for inference)
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load model or processor: {e}")
        print("Please ensure you have the necessary libraries (transformers, torch, PIL, accelerate).")
        print("If the model is gated or private, you might need to log in via `huggingface-cli login`.")
        print("Also, ensure your hardware and PyTorch installation support the chosen dtype (e.g., bfloat16 often requires newer GPUs).")
        return # Exit if model loading fails

    # --- Process Input Path ---
    input_path = args.input_path

    if os.path.isfile(input_path):
        # Input is a single file
        if input_path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            caption = generate_caption_for_image(input_path, PROCESSOR, MODEL)
            if caption: # Ensure caption is not None
                save_caption_to_file(caption, input_path)
        else:
            print(f"Skipping non-image file (or unsupported extension): {input_path}")
    elif os.path.isdir(input_path):
        # Input is a directory
        print(f"Processing images in folder: {input_path}")
        image_files_found = 0
        for item_name in os.listdir(input_path):
            item_path = os.path.join(input_path, item_name)
            if os.path.isfile(item_path) and item_path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_files_found += 1
                caption = generate_caption_for_image(item_path, PROCESSOR, MODEL)
                if caption: # Ensure caption is not None
                    save_caption_to_file(caption, item_path)
            elif os.path.isfile(item_path): # It's a file but not a supported image
                print(f"  Skipping non-image file (or unsupported extension): {item_name} in folder.")
        if image_files_found == 0:
            print(f"No supported image files found in directory: {input_path}")
    else:
        print(f"Error: Input path '{input_path}' is not a valid file or directory.")

    print("Processing complete.")

if __name__ == "__main__":
    main()
