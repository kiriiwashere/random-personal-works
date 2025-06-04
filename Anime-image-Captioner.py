import os
import argparse
from PIL import Image, UnidentifiedImageError, ImageTk
import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import threading
import requests # For downloading images from URL
from io import BytesIO
import time
import sys # To check for CLI arguments

# --- GUI Imports (only if not in CLI mode) ---
try:
    import tkinter as tk
    from tkinter import filedialog, ttk, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    # print("Tkinter not found. UI mode will not be available.")

# CLI run commands
# image to folder or image
# python Anime-image-Captioner.py /path/to/your/image.jpg
# python Anime-image-Captioner.py /path/to/your/folder/
# cuda device
# python Anime-image-Captioner /path/to/your/image.jpg --device cpu
# python Anime-image-Captioner /path/to/your/image.jpg --device cuda
# torch data
# python Anime-image-Captioner.py /path/to/your/image.jpg --dtype float16
# python Anime-image-Captioner.py /path/to/your/image.jpg --dtype float32
# output length
# python Anime-image-Captioner.py /path/to/image.jpg --max_tokens 768
# python Anime-image-Captioner.py /path/to/folder/ --max_tokens 1024

# --- Configuration ---
BASE_MODEL_ID = "Andres77872/SmolVLM-500M-anime-caption-v0.2"
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
DEFAULT_MAX_NEW_TOKENS = 1024 # Default value for max token output

# --- Model Components (Global) ---
PROCESSOR = None
MODEL = None
MODEL_LOADED = False
MODEL_LOADING_LOCK = threading.Lock() # To prevent multiple load attempts

# --- Custom Stopping Criteria ---
class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequence_str: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_sequence_str = stop_sequence_str

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        max_keep = len(self.stop_sequence_str) + 10
        text_to_check = generated_text[-max_keep:] if len(generated_text) > max_keep else generated_text
        return self.stop_sequence_str in text_to_check

# --- Input Preparation Function ---
def prepare_model_inputs(image: Image.Image, processor_instance, model_instance):
    question = "describe the image"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    max_image_edge = processor_instance.image_processor.max_image_size["longest_edge"]
    current_size_config = processor_instance.image_processor.size.copy()
    if "longest_edge" in current_size_config and current_size_config["longest_edge"] > max_image_edge:
        current_size_config["longest_edge"] = max_image_edge
    prompt_str = processor_instance.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor_instance(
        text=[prompt_str], images=[[image]], return_tensors='pt', padding=True, size=current_size_config
    )
    inputs = {k: v.to(model_instance.device) for k, v in inputs.items()}
    return inputs

# --- Image Processing and Caption Generation ---
def generate_caption_for_image_core(image_obj: Image.Image, processor_instance, model_instance, 
                                    max_new_tokens_val=DEFAULT_MAX_NEW_TOKENS, 
                                    ui_update_callback=None, image_name_for_log="image"):
    #Core logic for generating a caption for a given PIL Image object.
    model_inputs = prepare_model_inputs(image_obj, processor_instance, model_instance)
    stop_sequence_str = "</RATING>"
    streamer = TextIteratorStreamer(
        processor_instance.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    custom_stopping_criteria = StoppingCriteriaList([
        StopOnTokens(processor_instance.tokenizer, stop_sequence_str)
    ])
    generation_kwargs = dict(
        **model_inputs, 
        streamer=streamer, 
        do_sample=False, 
        max_new_tokens=max_new_tokens_val, # Use the passed or default value
        pad_token_id=processor_instance.tokenizer.pad_token_id,
        stopping_criteria=custom_stopping_criteria,
    )

    generation_thread = threading.Thread(target=model_instance.generate, kwargs=generation_kwargs)
    generation_thread.start()

    caption_parts = []
    if not ui_update_callback: # If no UI callback, print to console (CLI mode)
        print(f"  Caption for {image_name_for_log} (max_tokens: {max_new_tokens_val}): ", end="")

    for new_text_chunk in streamer:
        if ui_update_callback:
            ui_update_callback(new_text_chunk)
        else:
            print(new_text_chunk, end="", flush=True)
        caption_parts.append(new_text_chunk)
    
    if not ui_update_callback:
        print()

    generation_thread.join()
    full_caption = "".join(caption_parts).strip()
    
    # Remove the stop_sequence string if it's part of the output
    if full_caption.endswith(stop_sequence_str):
        full_caption = full_caption[:-len(stop_sequence_str)].strip()
        
    return full_caption

# --- File Handling ---
def save_caption_to_file(caption: str, image_file_path: str, is_url_image=False):
    if caption is None or not caption.strip():
        print(f"  Skipping save for {image_file_path} due to empty or no caption.")
        return None 

    if is_url_image:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_url_name = os.path.basename(image_file_path).split('?')[0] 
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        simple_url_name = ''.join(c for c in base_url_name if c in valid_chars)[:50]
        output_filename = f"url_image_{simple_url_name}_{timestamp}.txt"
        output_txt_path = os.path.join(os.getcwd(), output_filename) 
    else:
        base_name, _ = os.path.splitext(image_file_path)
        output_txt_path = base_name + ".txt"
    
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"  Caption successfully saved to: {output_txt_path}")
        return output_txt_path 
    except Exception as e:
        print(f"  Error saving caption to {output_txt_path}: {e}")
        return None

# --- Model Loading Function ---
def load_model_and_processor(device="auto", dtype_str="bfloat16", status_update_func=None):
    global PROCESSOR, MODEL, MODEL_LOADED
    
    with MODEL_LOADING_LOCK: 
        if MODEL_LOADED:
            if status_update_func: status_update_func("Model already loaded.")
            return True

        if status_update_func: status_update_func(f"Loading model: {BASE_MODEL_ID}...")
        print(f"Loading model: {BASE_MODEL_ID} (Device: {device}, Dtype: {dtype_str})...")

        if dtype_str == "bfloat16": torch_dtype = torch.bfloat16
        elif dtype_str == "float16": torch_dtype = torch.float16
        else: torch_dtype = torch.float32
        
        try:
            PROCESSOR = AutoProcessor.from_pretrained(BASE_MODEL_ID)
            MODEL = Idefics3ForConditionalGeneration.from_pretrained(
                BASE_MODEL_ID, device_map=device, torch_dtype=torch_dtype
            )
            MODEL.eval()
            MODEL_LOADED = True
            if status_update_func: status_update_func("Model and processor loaded successfully.")
            print("Model and processor loaded successfully.")
            return True
        except Exception as e:
            error_msg = f"Fatal Error: Could not load model or processor: {e}"
            if status_update_func: status_update_func(error_msg)
            print(error_msg)
            return False

# --- Tkinter Application Class ---
# We dont touch anything below this point
if GUI_AVAILABLE:
    class CaptionApp:
        def __init__(self, root_window):
            self.root = root_window
            self.root.title("Image Captioner")
            self.root.geometry("700x800") # Increased height for new field

            # --- UI Frames ---
            control_frame = ttk.Frame(self.root, padding="10")
            control_frame.pack(fill=tk.X)

            image_preview_frame = ttk.LabelFrame(self.root, text="Image Preview", padding="10")
            image_preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            output_frame = ttk.LabelFrame(self.root, text="Generated Caption", padding="10")
            output_frame.pack(fill=tk.X, padx=10, pady=5)
            
            status_frame = ttk.Frame(self.root, padding="10")
            status_frame.pack(fill=tk.X)

            # --- Controls ---
            # File Selection
            ttk.Button(control_frame, text="Select Image File", command=self.select_image_file).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            self.file_path_label = ttk.Label(control_frame, text="No file selected")
            self.file_path_label.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="w") # Span 3 for new layout

            # Folder Selection
            ttk.Button(control_frame, text="Select Folder", command=self.select_folder).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
            self.folder_path_label = ttk.Label(control_frame, text="No folder selected")
            self.folder_path_label.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")

            # URL Input
            ttk.Label(control_frame, text="Image URL:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
            self.url_entry = ttk.Entry(control_frame, width=50)
            self.url_entry.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

            # Max New Tokens Input
            ttk.Label(control_frame, text="Max New Tokens:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
            self.max_tokens_var = tk.StringVar(value=str(DEFAULT_MAX_NEW_TOKENS))
            self.max_tokens_entry = ttk.Entry(control_frame, textvariable=self.max_tokens_var, width=10)
            self.max_tokens_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")


            # Generate Button
            self.generate_button = ttk.Button(control_frame, text="Generate Caption(s)", command=self.start_captioning_process)
            self.generate_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10) # Adjusted row
            
            control_frame.columnconfigure(1, weight=1) 

            # --- Image Preview ---
            self.image_preview_label = ttk.Label(image_preview_frame, text="Preview will appear here")
            self.image_preview_label.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)
            self.current_pil_image = None 
            self.current_tk_image = None 


            # --- Output Area ---
            self.caption_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10, state=tk.DISABLED) # Increased height
            self.caption_text.pack(fill=tk.X, expand=True, padx=5, pady=5)

            # --- Status Bar & Progress Bar ---
            self.status_label = ttk.Label(status_frame, text="Ready. Select an image, folder, or enter a URL.")
            self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", length=200, mode="determinate")
            self.progress_bar.pack(side=tk.RIGHT, padx=5)

            self.selected_file_path = None
            self.selected_folder_path = None

        def _get_max_tokens_from_ui(self):
            try:
                val = int(self.max_tokens_var.get())
                if val <= 0: return DEFAULT_MAX_NEW_TOKENS # Basic validation
                return val
            except ValueError:
                return DEFAULT_MAX_NEW_TOKENS # Fallback if not a valid integer

        def _update_status(self, message):
            self.status_label.config(text=message)
            self.root.update_idletasks()

        def _update_caption_display(self, chunk):
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.insert(tk.END, chunk)
            self.caption_text.see(tk.END) 
            self.caption_text.config(state=tk.DISABLED)
            self.root.update_idletasks()
        
        def _clear_caption_display(self):
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.config(state=tk.DISABLED)

        def _clear_image_preview(self):
            self.image_preview_label.config(image='', text="Preview will appear here")
            self.current_pil_image = None
            self.current_tk_image = None 

        def _display_image_preview(self, image_path_or_pil_obj):
            self._clear_image_preview()
            try:
                if isinstance(image_path_or_pil_obj, str): 
                    pil_img = Image.open(image_path_or_pil_obj).convert("RGB")
                elif isinstance(image_path_or_pil_obj, Image.Image): 
                    pil_img = image_path_or_pil_obj
                else:
                    self._update_status("Error: Invalid image source for preview.")
                    return

                self.current_pil_image = pil_img 
                max_width = self.image_preview_label.winfo_width() if self.image_preview_label.winfo_width() > 10 else 400
                max_height = self.image_preview_label.winfo_height() if self.image_preview_label.winfo_height() > 10 else 300
                img_copy = pil_img.copy() 
                img_copy.thumbnail((max_width - 20, max_height - 20), Image.Resampling.LANCZOS) 
                self.current_tk_image = ImageTk.PhotoImage(img_copy) 
                self.image_preview_label.config(image=self.current_tk_image, text="")
                self.root.update_idletasks()
            except Exception as e:
                self._update_status(f"Preview Error: {e}")
                print(f"Image preview error: {e}")


        def select_image_file(self):
            self._clear_image_preview()
            filepath = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*"))
            )
            if filepath:
                self.selected_file_path = filepath
                self.selected_folder_path = None 
                self.url_entry.delete(0, tk.END) 
                self.file_path_label.config(text=os.path.basename(filepath))
                self.folder_path_label.config(text="No folder selected")
                self._update_status(f"Selected file: {os.path.basename(filepath)}")
                self._display_image_preview(filepath)

        def select_folder(self):
            self._clear_image_preview()
            folderpath = filedialog.askdirectory(title="Select Folder Containing Images")
            if folderpath:
                self.selected_folder_path = folderpath
                self.selected_file_path = None 
                self.url_entry.delete(0, tk.END) 
                self.folder_path_label.config(text=os.path.basename(folderpath))
                self.file_path_label.config(text="No file selected")
                self._update_status(f"Selected folder: {os.path.basename(folderpath)}")
                first_image = next((os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)), None)
                if first_image:
                    self._display_image_preview(first_image)
                else:
                    self._update_status(f"Selected folder: {os.path.basename(folderpath)}. No previewable images found.")


        def start_captioning_process(self):
            if not MODEL_LOADED:
                self._update_status("Model not loaded. Attempting to load...")
                self.generate_button.config(state=tk.DISABLED)
                threading.Thread(target=self._load_model_and_then_caption, daemon=True).start()
            else:
                self._initiate_caption_generation_thread()
        
        def _load_model_and_then_caption(self):
            success = load_model_and_processor(device="auto", dtype_str="bfloat16", status_update_func=self._update_status)
            self.root.after(0, self.generate_button.config, {"state": tk.NORMAL}) 
            if success:
                self.root.after(0, self._initiate_caption_generation_thread) 
            else:
                self.root.after(0, self._update_status, "Model loading failed. Cannot generate captions.")


        def _initiate_caption_generation_thread(self):
            self.generate_button.config(state=tk.DISABLED)
            self._clear_caption_display()
            
            url_text = self.url_entry.get().strip()
            max_tokens = self._get_max_tokens_from_ui()

            if self.selected_file_path:
                target_path = self.selected_file_path
                mode = "file"
            elif self.selected_folder_path:
                target_path = self.selected_folder_path
                mode = "folder"
            elif url_text:
                target_path = url_text
                mode = "url"
            else:
                self._update_status("Please select an image file, folder, or enter a URL.")
                self.generate_button.config(state=tk.NORMAL)
                return

            threading.Thread(target=self._process_target, args=(target_path, mode, max_tokens), daemon=True).start()

        def _process_target(self, target_path, mode, max_tokens_val):
            # Handles the actual processing in a separate thread.
            try:
                if mode == "file":
                    self._process_single_image_ui(target_path, max_tokens_val=max_tokens_val)
                elif mode == "folder":
                    self._process_folder_ui(target_path, max_tokens_val=max_tokens_val)
                elif mode == "url":
                    self._process_url_ui(target_path, max_tokens_val=max_tokens_val)
            finally:
                self.root.after(0, self.generate_button.config, {"state": tk.NORMAL})
                self.root.after(0, self._update_status, "Processing complete. Ready for next task.")


        def _process_single_image_ui(self, image_path_or_obj, is_url=False, original_url=None, max_tokens_val=DEFAULT_MAX_NEW_TOKENS):
            log_name = original_url if is_url else os.path.basename(image_path_or_obj)
            self.root.after(0, self._update_status, f"Processing: {log_name} (max_tokens: {max_tokens_val})...")
            self.root.after(0, self.progress_bar.config, {"value": 0, "maximum": 100, "mode": "indeterminate"})
            self.root.after(0, self.progress_bar.start)
            
            try:
                if is_url: 
                    img_obj = image_path_or_obj 
                    self.root.after(0, self._display_image_preview, img_obj) 
                else: 
                    img_obj = Image.open(image_path_or_obj).convert("RGB")
                    self.root.after(0, self._display_image_preview, image_path_or_obj) 

            except FileNotFoundError:
                self.root.after(0, self._update_status, f"Error: Image file not found: {log_name}")
                return
            except UnidentifiedImageError:
                self.root.after(0, self._update_status, f"Error: Cannot identify image: {log_name}")
                return
            except Exception as e:
                self.root.after(0, self._update_status, f"Error loading image {log_name}: {e}")
                return
            
            self.root.after(0, self._clear_caption_display) 
            caption = generate_caption_for_image_core(
                img_obj, PROCESSOR, MODEL, 
                max_new_tokens_val=max_tokens_val,
                ui_update_callback=lambda chunk: self.root.after(0, self._update_caption_display, chunk),
                image_name_for_log=log_name
            )

            if caption:
                saved_path = save_caption_to_file(caption, original_url if is_url else image_path_or_obj, is_url_image=is_url)
                if saved_path:
                    self.root.after(0, self._update_status, f"Caption for {log_name} saved to {os.path.basename(saved_path)}")
                else:
                    self.root.after(0, self._update_status, f"Caption generated for {log_name}, but failed to save.")

            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, self.progress_bar.config, {"value": 0, "mode": "determinate"})


        def _process_folder_ui(self, folder_path, max_tokens_val=DEFAULT_MAX_NEW_TOKENS):
            self.root.after(0, self._update_status, f"Processing folder: {os.path.basename(folder_path)}...")
            image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
            ]
            if not image_files:
                self.root.after(0, self._update_status, "No supported image files found in the folder.")
                return

            self.root.after(0, self.progress_bar.config, {"value": 0, "maximum": len(image_files), "mode": "determinate"})
            
            for i, image_file_path in enumerate(image_files):
                self.root.after(0, self._clear_caption_display) 
                self.root.after(0, self._update_status, f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_file_path)}")
                
                self._process_single_image_ui(image_file_path, is_url=False, max_tokens_val=max_tokens_val) 
                
                self.root.after(0, self.progress_bar.config, {"value": i + 1})

            self.root.after(0, self._update_status, f"Folder processing complete for {os.path.basename(folder_path)}.")


        def _process_url_ui(self, image_url, max_tokens_val=DEFAULT_MAX_NEW_TOKENS):
            self.root.after(0, self._update_status, f"Downloading image from URL: {image_url[:50]}...")
            self.root.after(0, self.progress_bar.config, {"value": 0, "maximum": 100, "mode": "indeterminate"})
            self.root.after(0, self.progress_bar.start)
            self.root.after(0, self._clear_image_preview) 

            try:
                response = requests.get(image_url, stream=True, timeout=10)
                response.raise_for_status() 
                content_type = response.headers.get('content-type')
                if content_type and not content_type.lower().startswith('image/'):
                    raise ValueError(f"URL does not point to a direct image (Content-Type: {content_type})")
                img_bytes = response.content
                img_obj = Image.open(BytesIO(img_bytes)).convert("RGB")
                self._process_single_image_ui(img_obj, is_url=True, original_url=image_url, max_tokens_val=max_tokens_val)
            except requests.exceptions.RequestException as e:
                self.root.after(0, self._update_status, f"URL Error: {e}")
            except UnidentifiedImageError:
                self.root.after(0, self._update_status, "URL Error: Content is not a valid image format.")
            except ValueError as e: 
                self.root.after(0, self._update_status, f"URL Error: {e}")
            except Exception as e:
                self.root.after(0, self._update_status, f"Error processing URL {image_url[:50]}...: {e}")
            finally:
                self.root.after(0, self.progress_bar.stop)
                self.root.after(0, self.progress_bar.config, {"value": 0, "mode": "determinate"})

# --- Main Execution Logic (CLI or UI) ---
def main_cli(cli_args):
    """Handles Command Line Interface operations."""
    if not load_model_and_processor(cli_args.device, cli_args.dtype):
        return 

    input_path = cli_args.input_path
    max_tokens = cli_args.max_tokens
    processed_count = 0

    if os.path.isfile(input_path):
        if input_path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            print(f"Processing single image file: {input_path}")
            try:
                img = Image.open(input_path).convert("RGB")
                caption = generate_caption_for_image_core(
                    img, PROCESSOR, MODEL, 
                    max_new_tokens_val=max_tokens,
                    image_name_for_log=os.path.basename(input_path)
                )
                if caption:
                    save_caption_to_file(caption, input_path)
                    processed_count +=1
            except Exception as e:
                print(f"  Error processing image {input_path}: {e}")
        else:
            print(f"Skipping non-image file (or unsupported extension): {input_path}")
    elif os.path.isdir(input_path):
        print(f"Processing images in folder: {input_path}")
        image_files_found = 0
        for item_name in os.listdir(input_path):
            item_path = os.path.join(input_path, item_name)
            if os.path.isfile(item_path) and item_path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_files_found += 1
                print(f"\nProcessing image {image_files_found}: {item_name}")
                try:
                    img = Image.open(item_path).convert("RGB")
                    caption = generate_caption_for_image_core(
                        img, PROCESSOR, MODEL, 
                        max_new_tokens_val=max_tokens,
                        image_name_for_log=item_name
                    )
                    if caption:
                        save_caption_to_file(caption, item_path)
                        processed_count +=1
                except Exception as e:
                    print(f"  Error processing image {item_path}: {e}")
            elif os.path.isfile(item_path):
                print(f"  Skipping non-image file (or unsupported extension): {item_name} in folder.")
        if image_files_found == 0:
            print(f"No supported image files found in directory: {input_path}")
    else:
        print(f"Error: Input path '{input_path}' is not a valid file or directory.")
    print(f"\nProcessing complete. {processed_count} caption(s) generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images. Runs GUI if no args.")
    parser.add_argument("input_path", type=str, nargs='?', default=None, help="Path to an image file or folder (CLI mode).")
    parser.add_argument("--device", type=str, default="auto", help="Device for model: 'cuda', 'cpu', 'auto' (CLI mode).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch dtype (CLI mode).")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help=f"Maximum number of new tokens to generate for caption (CLI mode). Default: {DEFAULT_MAX_NEW_TOKENS}")
    
    args = parser.parse_args()

    if args.input_path: 
        print("Running in Command-Line Interface (CLI) mode.")
        main_cli(args)
    elif GUI_AVAILABLE:
        print("No input path provided via CLI. Attempting to launch GUI mode...")
        root = tk.Tk()
        app = CaptionApp(root)
        root.mainloop()
    else:
        print("GUI is unavailable (Tkinter might be missing or display not found) and no input_path provided for CLI mode.")
        parser.print_help()
