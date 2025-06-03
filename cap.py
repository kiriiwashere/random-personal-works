# pip install torch transformers pillow
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Initialize device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Andres77872/SmolVLM-500M-anime-caption-v0.2")
model = AutoModelForVision2Seq.from_pretrained("Andres77872/SmolVLM-500M-anime-caption-v0.2").to(device)

def generate_caption(image_path):
    """Generate a caption for a single image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption.strip()

def caption_images_in_folder(folder_path):
    """Generate captions for all images in the specified folder."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                caption = generate_caption(image_path)
                print(f"Caption for {filename}: {caption}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    folder_path = "path_to_your_image_folder"  # Replace with your folder path
    caption_images_in_folder(folder_path)
