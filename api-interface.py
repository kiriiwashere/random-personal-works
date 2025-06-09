import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
import base64
import io
import threading
import json
import mimetypes

def call_openrouter_api(api_key, model, prompt, image_data_base64, max_tokens, http_referer, x_title):
    """
    Calls the OpenRouter API for multimodal chat completions.
    """
    try:
        mime_type = "image/jpeg"
        if image_data_base64.startswith('iVBORw0KGgo='):
             mime_type = "image/png"
        elif image_data_base64.startswith('/9j/'):
             mime_type = "image/jpeg"

        image_url = f"data:{mime_type};base64,{image_data_base64}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title

        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )

        response.raise_for_status()  # raise an exception for bad status codes (4xx or 5xx)
        return response.json()

    except requests.exceptions.RequestException as e:
        # handle network-related errors
        error_message = f"Network Error: {e}"
        print(error_message)
        return {"error": {"message": error_message}}
    except Exception as e:
        # handle other unexpected errors
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return {"error": {"message": error_message}}

class CaptionGeneratorApp(tk.Tk):


    def __init__(self):
        super().__init__()
        self.title("API Image Caption Interface (OpenRouter)")
        self.geometry("900x700")
        self.configure(bg="#2c3e50")

        # style vodoo
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TFrame", background="#2c3e50")
        style.configure("TLabel", background="#2c3e50", foreground="#ecf0f1", font=("Helvetica", 11))
        style.configure("TButton", background="#3498db", foreground="#ffffff", font=("Helvetica", 10, "bold"), borderwidth=0)
        style.map("TButton", background=[('active', '#2980b9')])
        style.configure("TCombobox", fieldbackground="#34495e", background="#34495e", foreground="#ecf0f1", arrowcolor="#ecf0f1")
        style.configure("TEntry", fieldbackground="#34495e", foreground="#ecf0f1", borderwidth=1, insertbackground="#ecf0f1")

        # states
        self.generation_thread = None
        self.stop_generation = threading.Event()
        self.image_data_base64 = None

        # frame
        main_frame = ttk.Frame(self, padding="20 20 20 20")
        main_frame.pack(expand=True, fill="both")

        # layout
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)

        # UI

        # api inout
        ttk.Label(main_frame, text="OpenRouter API Key:").grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(main_frame, textvariable=self.api_key_var, width=50, show="*")
        self.api_key_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=10, sticky="ew")

        # model select
        ttk.Label(main_frame, text="Select Model:").grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.model_var = tk.StringVar(value="google/gemini-pro-vision")
        self.model_select = ttk.Combobox(
            main_frame,
            textvariable=self.model_var,
            values=[
                "google/gemini-2.0-flash-exp:free",
                "google/gemma-3-4b-it:free",
                "openai/gpt-4o",
                "openai/gpt-4-vision-preview",
                "anthropic/claude-3-haiku:beta",
                "meta-llama/llama-4-maverick:free",
            ],
            state="readonly"
        )
        self.model_select.grid(row=1, column=1, columnspan=2, padx=5, pady=10, sticky="ew")

        # prompt input
        ttk.Label(main_frame, text="Prompt:").grid(row=2, column=0, padx=5, pady=10, sticky="nw")
        self.prompt_text = tk.Text(main_frame, height=4, width=50, wrap="word", bg="#34495e", fg="#ecf0f1", insertbackground="#ecf0f1", relief="flat", borderwidth=2)
        self.prompt_text.insert("1.0", "Describe this image in a detailed and vivid manner.")
        self.prompt_text.grid(row=2, column=1, columnspan=2, padx=5, pady=10, sticky="ew")

        # image input
        ttk.Label(main_frame, text="Image Link/File:").grid(row=3, column=0, padx=5, pady=10, sticky="w")
        self.image_path_var = tk.StringVar()
        self.image_path_entry = ttk.Entry(main_frame, textvariable=self.image_path_var, width=50)
        self.image_path_entry.grid(row=3, column=1, padx=5, pady=10, sticky="ew")

        self.browse_button = ttk.Button(main_frame, text="Browse...", command=self.browse_file)
        self.browse_button.grid(row=3, column=2, padx=5, pady=10)
        self.image_path_entry.bind('<KeyRelease>', self.handle_path_change)

        # image preview
        # still a major problem with the image preview. pushes other elements down out of bounds
        self.image_preview_label = ttk.Label(main_frame, text="Image Preview", background="#34495e", anchor="center")
        self.image_preview_label.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
        main_frame.rowconfigure(4, weight=1) 

        # 6. adv option
        adv_frame = ttk.Frame(main_frame)
        adv_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(10,0))
        adv_frame.columnconfigure(1, weight=1)
        adv_frame.columnconfigure(3, weight=1)

        ttk.Label(adv_frame, text="Max Tokens:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_tokens_var = tk.StringVar(value="1024")
        self.max_tokens_entry = ttk.Entry(adv_frame, textvariable=self.max_tokens_var, width=10)
        self.max_tokens_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(adv_frame, text="HTTP Referer:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.http_referer_var = tk.StringVar(value="https://github.com/your-repo")
        self.http_referer_entry = ttk.Entry(adv_frame, textvariable=self.http_referer_var)
        self.http_referer_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(adv_frame, text="X-Title:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.x_title_var = tk.StringVar(value="My AI GUI")
        self.x_title_entry = ttk.Entry(adv_frame, textvariable=self.x_title_var)
        self.x_title_entry.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        # 7. ai output
        ttk.Label(main_frame, text="AI Output:").grid(row=6, column=0, padx=5, pady=10, sticky="nw")
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=6, column=1, columnspan=2, padx=5, pady=10, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output_text = tk.Text(
            output_frame, height=5, width=50, wrap="word", state="disabled",
            bg="#1c2833", fg="#ecf0f1", relief="flat", borderwidth=2
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")

        output_scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        output_scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.config(yscrollcommand=output_scrollbar.set)

        # 8. buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=20, sticky="e")

        self.generate_button = ttk.Button(button_frame, text="Generate", command=self.start_generation)
        self.generate_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_generation_thread, state="disabled")
        self.stop_button.pack(side="left", padx=5)

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*"))
        )
        if filepath:
            self.image_path_var.set(filepath)
            self.load_image(filepath)

    def handle_path_change(self, event):
        path = self.image_path_var.get()
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            if path.startswith(('http://', 'https://')):
                response = requests.get(path, stream=True)
                response.raise_for_status()
                image_bytes = response.content
            else:
                with open(path, "rb") as f:
                    image_bytes = f.read()

            self.image_data_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image = Image.open(io.BytesIO(image_bytes))
            max_size = (400, 250)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_preview_label.config(image=self.photo, text="")
        except Exception as e:
            self.image_preview_label.config(image=None, text=f"Error: Could not load image.\n{e}")
            self.image_data_base64 = None

    def set_output_text(self, text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)
        self.output_text.config(state="disabled")

    def start_generation(self):
        if not self.api_key_var.get():
            messagebox.showerror("Error", "Please enter your OpenRouter API key.")
            return
        if not self.image_data_base64:
            messagebox.showerror("Error", "Please provide a valid image first.")
            return
        if not self.prompt_text.get("1.0", "end-1c").strip():
            messagebox.showerror("Error", "Prompt cannot be empty.")
            return

        self.stop_generation.clear()
        self.generate_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.set_output_text("Generating caption...")

        self.generation_thread = threading.Thread(
            target=self.run_generation_task,
            daemon=True
        )
        self.generation_thread.start()

    def stop_generation_thread(self):
        self.stop_generation.set()
        self.set_output_text("Generation stopped by user.")
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def run_generation_task(self):
        try:
            api_key = self.api_key_var.get()
            prompt = self.prompt_text.get("1.0", "end-1c")
            model = self.model_var.get()
            max_tokens = int(self.max_tokens_var.get())
            http_referer = self.http_referer_var.get()
            x_title = self.x_title_var.get()

            if self.stop_generation.is_set(): return
            
            response_data = call_openrouter_api(api_key, model, prompt, self.image_data_base64, max_tokens, http_referer, x_title)

            if self.stop_generation.is_set(): return
            
            # api error handle
            if 'error' in response_data:
                error_msg = response_data['error'].get('message', 'Unknown API error.')
                self.after(0, self.set_output_text, f"API Error: {error_msg}")
                return

            caption = response_data.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not parse caption from response.")
            self.after(0, self.set_output_text, caption)

        except Exception as e:
            self.after(0, self.set_output_text, f"An error occurred: {e}")
        finally:
            def re_enable_buttons():
                self.generate_button.config(state="normal")
                self.stop_button.config(state="disabled")
            self.after(0, re_enable_buttons)


if __name__ == "__main__":
    app = CaptionGeneratorApp()
    app.mainloop()
