# szuru_uploader_gui.py

import argparse
import subprocess
import requests
import json
import os
import logging
import sys
import threading
import queue # For thread-safe communication with GUI

# --- GUI Imports (only if not in CLI mode) ---
try:
    import tkinter as tk
    from tkinter import filedialog, ttk, scrolledtext, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# --- Configuration (Defaults, can be overridden by GUI/CLI/Env Vars) ---
SZURUBOORU_API_URL = os.environ.get("SZURUBOORU_API_URL", "http://localhost:8080/api")
SZURUBOORU_USERNAME = os.environ.get("SZURUBOORU_USERNAME", "")
SZURUBOORU_API_KEY = os.environ.get("SZURUBOORU_API_KEY", "")

CAPTIONER_SCRIPT_FILENAME = "Anime-image-Captioner.py" # Make sure this is correct
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAPTIONER_SCRIPT_PATH = os.path.join(SCRIPT_DIR, CAPTIONER_SCRIPT_FILENAME)

DEFAULT_CAPTIONER_PYTHON_EXECUTABLE = sys.executable
DEFAULT_CAPTIONER_DEVICE = "auto"
DEFAULT_CAPTIONER_DTYPE = "bfloat16"
DEFAULT_CAPTIONER_MAX_TOKENS = 1024
DEFAULT_AI_VERSION_TAG = "1.0"

# --- Logging Setup ---
# We'll add a Tkinter handler for the GUI later
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
                    handlers=[logging.FileHandler("szuru_uploader.log")])
# Console handler for CLI mode will be added if GUI is not run.
# If GUI runs, we'll add a handler to put logs into a ScrolledText widget.


# --- [ PREVIOUSLY DEFINED CORE FUNCTIONS: ] ---
# parse_captioner_output(raw_output_str: str) -> dict
# run_captioning_model(...) -> dict | None
# class SzurubooruClient: ...
# calculate_md5(file_path)
# process_single_image(...) -> bool
# --- [ THESE FUNCTIONS REMAIN LARGELY THE SAME AS IN THE PREVIOUS szuru_uploader.py ] ---
# --- [ For brevity, I will paste them at the end of this script block ] ---
# --- [ Assume they are present here ] ---


# --- Tkinter GUI Application Class ---
if GUI_AVAILABLE:
    class TextHandler(logging.Handler):
        """This class allows you to log to a Tkinter Text or ScrolledText widget."""
        def __init__(self, text_widget):
            logging.Handler.__init__(self)
            self.text_widget = text_widget
            self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))


        def emit(self, record):
            msg = self.format(record)
            def append():
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.configure(state='disabled')
                self.text_widget.yview(tk.END) # Auto-scroll
            # This makes the update thread-safe by scheduling it in the Tkinter main loop
            self.text_widget.after(0, append)

    class UploaderApp:
        def __init__(self, root_window):
            self.root = root_window
            self.root.title("Szurubooru AI Uploader")
            self.root.geometry("800x900") # Adjusted size

            self.processing_thread = None
            self.input_queue = queue.Queue() # To send image paths to the processing thread

            # --- Main PanedWindow for resizable sections ---
            main_paned_window = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
            main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # --- Top Frame for controls ---
            controls_outer_frame = ttk.Frame(main_paned_window) # No need for LabelFrame here
            main_paned_window.add(controls_outer_frame, weight=0) # Smaller weight, less resizable space


            # Create a Notebook for tabbed settings
            self.notebook = ttk.Notebook(controls_outer_frame)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # --- Tab 1: Szurubooru & Input ---
            tab1 = ttk.Frame(self.notebook, padding="10")
            self.notebook.add(tab1, text='Szurubooru & Input')
            self._create_szuru_input_tab(tab1)

            # --- Tab 2: Processing Options ---
            tab2 = ttk.Frame(self.notebook, padding="10")
            self.notebook.add(tab2, text='Processing Options')
            self._create_processing_options_tab(tab2)

            # --- Tab 3: AI Captioner Settings ---
            tab3 = ttk.Frame(self.notebook, padding="10")
            self.notebook.add(tab3, text='AI Captioner Settings')
            self._create_captioner_settings_tab(tab3)

            # --- Action Buttons Frame ---
            action_frame = ttk.Frame(controls_outer_frame, padding="5")
            action_frame.pack(fill=tk.X, pady=5)

            self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing_gui)
            self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

            self.stop_button = ttk.Button(action_frame, text="Stop Processing", command=self.stop_processing_gui, state=tk.DISABLED)
            self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)


            # --- Log Output Frame (Bottom part of PanedWindow) ---
            log_frame_outer = ttk.LabelFrame(main_paned_window, text="Logs & Status", padding="10")
            main_paned_window.add(log_frame_outer, weight=1) # Larger weight, more resizable

            self.log_text = scrolledtext.ScrolledText(log_frame_outer, wrap=tk.WORD, height=15, state=tk.DISABLED)
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # --- Status Bar ---
            status_bar_frame = ttk.Frame(log_frame_outer, padding="5") # Inside log_frame_outer
            status_bar_frame.pack(fill=tk.X, side=tk.BOTTOM)

            self.status_label = ttk.Label(status_bar_frame, text="Ready. Configure settings and select images/folders.")
            self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.progress_bar = ttk.Progressbar(status_bar_frame, orient="horizontal", length=200, mode="determinate")
            self.progress_bar.pack(side=tk.RIGHT, padx=5)

            # Setup logging to the GUI
            gui_log_handler = TextHandler(self.log_text)
            logging.getLogger().addHandler(gui_log_handler)
            logging.getLogger().setLevel(logging.INFO) # Ensure root logger level is appropriate

            self.stop_processing_flag = threading.Event()


        def _create_szuru_input_tab(self, parent_tab):
            # --- Szurubooru Connection Details ---
            szuru_frame = ttk.LabelFrame(parent_tab, text="Szurubooru Connection", padding="10")
            szuru_frame.pack(fill=tk.X, expand=True, pady=5)

            ttk.Label(szuru_frame, text="API URL:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.szuru_url_var = tk.StringVar(value=SZURUBOORU_API_URL)
            ttk.Entry(szuru_frame, textvariable=self.szuru_url_var, width=60).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

            ttk.Label(szuru_frame, text="Username:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.szuru_user_var = tk.StringVar(value=SZURUBOORU_USERNAME)
            ttk.Entry(szuru_frame, textvariable=self.szuru_user_var, width=40).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

            ttk.Label(szuru_frame, text="API Token:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.szuru_token_var = tk.StringVar(value=SZURUBOORU_API_KEY)
            ttk.Entry(szuru_frame, textvariable=self.szuru_token_var, width=40, show="*").grid(row=2, column=1, padx=5, pady=2, sticky="ew")
            szuru_frame.columnconfigure(1, weight=1)

            # --- Input Selection ---
            input_frame = ttk.LabelFrame(parent_tab, text="Image Input", padding="10")
            input_frame.pack(fill=tk.X, expand=True, pady=10)

            self.selected_paths_listbox = tk.Listbox(input_frame, selectmode=tk.EXTENDED, height=5)
            self.selected_paths_listbox.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5, pady=5)

            btn_frame = ttk.Frame(input_frame)
            btn_frame.pack(fill=tk.X)
            ttk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Clear List", command=self.clear_selection_list).pack(side=tk.LEFT, padx=5)

            self.recursive_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(input_frame, text="Process Subfolders Recursively (if folder added)", variable=self.recursive_var).pack(anchor="w", padx=5)


        def _create_processing_options_tab(self, parent_tab):
            proc_frame = ttk.LabelFrame(parent_tab, text="Upload & Update Rules", padding="10")
            proc_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            self.force_reupload_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(proc_frame, text="Force Re-upload (creates duplicates if MD5 matches)", variable=self.force_reupload_var).pack(anchor="w", padx=5, pady=2)

            self.force_update_existing_var = tk.BooleanVar(value=True) # Default to true for ease
            ttk.Checkbutton(proc_frame, text="Force Update Existing (if MD5 matches, update metadata)", variable=self.force_update_existing_var).pack(anchor="w", padx=5, pady=2)

            ttk.Label(proc_frame, text="AI Processed Version Tag:").pack(anchor="w", padx=5, pady=(10,0))
            self.ai_version_tag_var = tk.StringVar(value=DEFAULT_AI_VERSION_TAG)
            ttk.Entry(proc_frame, textvariable=self.ai_version_tag_var, width=15).pack(anchor="w", padx=5, pady=2)


        def _create_captioner_settings_tab(self, parent_tab):
            cap_frame = ttk.LabelFrame(parent_tab, text="AI Captioner Script Configuration", padding="10")
            cap_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            ttk.Label(cap_frame, text="Python Executable:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.cap_py_exe_var = tk.StringVar(value=DEFAULT_CAPTIONER_PYTHON_EXECUTABLE)
            ttk.Entry(cap_frame, textvariable=self.cap_py_exe_var, width=50).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

            ttk.Label(cap_frame, text="Captioner Script Path:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.cap_script_path_var = tk.StringVar(value=DEFAULT_CAPTIONER_SCRIPT_PATH)
            entry_script_path = ttk.Entry(cap_frame, textvariable=self.cap_script_path_var, width=50)
            entry_script_path.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
            ttk.Button(cap_frame, text="Browse", command=lambda: self.browse_file(self.cap_script_path_var, [("Python files", "*.py")])).grid(row=1, column=2, padx=5, pady=2)


            ttk.Label(cap_frame, text="Device:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.cap_device_var = tk.StringVar(value=DEFAULT_CAPTIONER_DEVICE)
            ttk.Combobox(cap_frame, textvariable=self.cap_device_var, values=["auto", "cpu", "cuda"], state="readonly").grid(row=2, column=1, padx=5, pady=2, sticky="w")

            ttk.Label(cap_frame, text="DType:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
            self.cap_dtype_var = tk.StringVar(value=DEFAULT_CAPTIONER_DTYPE)
            ttk.Combobox(cap_frame, textvariable=self.cap_dtype_var, values=["bfloat16", "float16", "float32"], state="readonly").grid(row=3, column=1, padx=5, pady=2, sticky="w")

            ttk.Label(cap_frame, text="Max New Tokens:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
            self.cap_max_tokens_var = tk.StringVar(value=str(DEFAULT_CAPTIONER_MAX_TOKENS))
            ttk.Entry(cap_frame, textvariable=self.cap_max_tokens_var, width=10).grid(row=4, column=1, padx=5, pady=2, sticky="w")
            cap_frame.columnconfigure(1, weight=1)


        def browse_file(self, string_var, file_types):
            filename = filedialog.askopenfilename(title="Select File", filetypes=file_types)
            if filename:
                string_var.set(filename)

        def add_files(self):
            filepaths = filedialog.askopenfilenames(
                title="Select Image Files",
                filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*"))
            )
            if filepaths:
                for fp in filepaths:
                    self.selected_paths_listbox.insert(tk.END, fp)
                self._update_status(f"{len(filepaths)} file(s) added to the list.")

        def add_folder(self):
            folderpath = filedialog.askdirectory(title="Select Folder Containing Images")
            if folderpath:
                # Add a special marker or just the path to distinguish folders if needed later
                # For now, just add it; processing logic will check if it's a dir.
                self.selected_paths_listbox.insert(tk.END, folderpath)
                self._update_status(f"Folder '{os.path.basename(folderpath)}' added to the list.")

        def clear_selection_list(self):
            self.selected_paths_listbox.delete(0, tk.END)
            self._update_status("Selection list cleared.")


        def _update_status(self, message):
            self.status_label.config(text=message)
            logging.info(message) # Also log to the text widget
            self.root.update_idletasks()

        def _set_processing_state(self, is_processing):
            state = tk.DISABLED if is_processing else tk.NORMAL
            self.start_button.config(state=state)
            self.stop_button.config(state=tk.NORMAL if is_processing else tk.DISABLED)
            # Disable input fields during processing
            for tab_index in range(self.notebook.index("end")): # Iterate through all tabs
                tab_widget = self.notebook.winfo_children()[tab_index]
                for widget in tab_widget.winfo_children(): # Iterate frames within tab
                    if isinstance(widget, (ttk.LabelFrame, ttk.Frame)):
                         for child in widget.winfo_children(): # Iterate widgets within frames
                            if isinstance(child, (ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Combobox, tk.Listbox)):
                                try: # Some widgets might not have 'state'
                                    child.config(state=state if child not in [self.start_button, self.stop_button] else child.cget('state'))
                                except tk.TclError:
                                    pass # Widget doesn't support state or already handled
                    elif isinstance(widget, (ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Combobox, tk.Listbox)):
                         widget.config(state=state)


        def start_processing_gui(self):
            if self.processing_thread and self.processing_thread.is_alive():
                messagebox.showwarning("Busy", "Processing is already in progress.")
                return

            self.stop_processing_flag.clear()
            self._set_processing_state(True)
            self._update_status("Starting processing...")

            # Collect all settings from GUI
            gui_args = self._collect_gui_args()
            if not gui_args: # Validation failed
                self._set_processing_state(False)
                return

            self.processing_thread = threading.Thread(target=self._processing_loop, args=(gui_args,), daemon=True)
            self.processing_thread.start()
            self.root.after(100, self.check_processing_thread)


        def stop_processing_gui(self):
            if self.processing_thread and self.processing_thread.is_alive():
                self._update_status("Stop signal sent. Finishing current image...")
                self.stop_processing_flag.set() # Signal the thread to stop
                self.stop_button.config(state=tk.DISABLED) # Prevent multiple clicks
            else:
                self._update_status("No active processing to stop.")

        def check_processing_thread(self):
            if self.processing_thread and self.processing_thread.is_alive():
                self.root.after(100, self.check_processing_thread) # Poll again
            else: # Thread finished or was never started properly
                self._set_processing_state(False)
                if not self.stop_processing_flag.is_set(): # if not stopped by user
                     self._update_status("Processing finished or was not started. Ready.")
                else: # if stopped by user
                     self._update_status("Processing stopped by user. Ready.")
                self.stop_processing_flag.clear()


        def _collect_gui_args(self):
            """Collects all settings from the GUI and returns them as an object or dict."""
            # Basic validation
            if not self.szuru_url_var.get() or not self.szuru_user_var.get() or not self.szuru_token_var.get():
                messagebox.showerror("Error", "Szurubooru API URL, Username, and Token are required.")
                return None
            if not self.cap_script_path_var.get() or not os.path.exists(self.cap_script_path_var.get()):
                messagebox.showerror("Error", "AI Captioner script path is invalid or file does not exist.")
                return None

            try:
                max_tokens = int(self.cap_max_tokens_var.get())
                if max_tokens <= 0: raise ValueError("Max tokens must be positive.")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid Max New Tokens value: {e}")
                return None

            # Create an object similar to argparse's Namespace
            class GuiArgs:
                pass
            args = GuiArgs()

            args.image_paths_from_gui = list(self.selected_paths_listbox.get(0, tk.END))
            if not args.image_paths_from_gui:
                messagebox.showerror("Error", "No image files or folders selected for processing.")
                return None

            args.recursive = self.recursive_var.get()
            args.force_reupload = self.force_reupload_var.get()
            args.force_update_existing = self.force_update_existing_var.get()
            args.ai_version_tag = self.ai_version_tag_var.get()

            args.szuru_url = self.szuru_url_var.get()
            args.szuru_user = self.szuru_user_var.get()
            args.szuru_token = self.szuru_token_var.get()

            args.captioner_py_exe = self.cap_py_exe_var.get()
            args.captioner_script_path = self.cap_script_path_var.get()
            args.captioner_device = self.cap_device_var.get()
            args.captioner_dtype = self.cap_dtype_var.get()
            args.captioner_max_tokens = max_tokens

            return args


        def _processing_loop(self, gui_args):
            """The main loop that runs in the worker thread."""
            self.root.after(0, self._update_status, "Processing thread started.")
            szuru_client = SzurubooruClient(gui_args.szuru_url, gui_args.szuru_user, gui_args.szuru_token)
            self.root.after(0, self.progress_bar.config, {"value": 0, "mode": "determinate"})

            all_image_files_to_process = []
            supported_extensions = tuple(ext.lower() for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'])

            for input_path_arg in gui_args.image_paths_from_gui:
                if self.stop_processing_flag.is_set(): break
                if os.path.isfile(input_path_arg):
                    if input_path_arg.lower().endswith(supported_extensions):
                        all_image_files_to_process.append(os.path.abspath(input_path_arg))
                    else:
                        self.root.after(0, self._update_status, f"Skipping non-supported file: {input_path_arg}")
                elif os.path.isdir(input_path_arg):
                    if gui_args.recursive:
                        for root_dir, _, files in os.walk(input_path_arg):
                            if self.stop_processing_flag.is_set(): break
                            for file in files:
                                if self.stop_processing_flag.is_set(): break
                                if file.lower().endswith(supported_extensions):
                                    all_image_files_to_process.append(os.path.join(root_dir, file))
                    else:
                        for file in os.listdir(input_path_arg):
                            if self.stop_processing_flag.is_set(): break
                            if file.lower().endswith(supported_extensions):
                                all_image_files_to_process.append(os.path.join(input_path_arg, file))
                else: # Should not happen if listbox only contains valid paths
                     self.root.after(0, self._update_status, f"Input path '{input_path_arg}' is not a valid file or directory. Skipping.")


            if not all_image_files_to_process and not self.stop_processing_flag.is_set():
                self.root.after(0, self._update_status, "No image files found to process based on selection.")
                return # Early exit from thread

            num_total_images = len(all_image_files_to_process)
            self.root.after(0, self.progress_bar.config, {"maximum": num_total_images if num_total_images > 0 else 1})

            successful_processing_count = 0
            failed_processing_count = 0

            for i, image_file in enumerate(all_image_files_to_process):
                if self.stop_processing_flag.is_set():
                    self.root.after(0, self._update_status, f"Processing stopped by user. Processed {i} of {num_total_images} images.")
                    break
                self.root.after(0, self._update_status, f"Processing image {i+1}/{num_total_images}: {os.path.basename(image_file)}")
                
                # process_single_image now uses gui_args directly
                if process_single_image(image_file, szuru_client, gui_args): # Pass gui_args which contains all settings
                    successful_processing_count += 1
                else:
                    failed_processing_count += 1
                self.root.after(0, self.progress_bar.config, {"value": i + 1})

            final_status = f"Finished. Success: {successful_processing_count}, Failed: {failed_processing_count}."
            if self.stop_processing_flag.is_set():
                final_status = f"Stopped. Processed: {successful_processing_count+failed_processing_count}, Success: {successful_processing_count}, Failed: {failed_processing_count}."
            self.root.after(0, self._update_status, final_status)
            # self.root.after(0, self._set_processing_state, False) # Handled by check_processing_thread

# --- Main Execution Logic (CLI or GUI) ---
def main_cli(cli_args):
    """Handles Command Line Interface operations."""
    # Add a console handler for CLI mode if not already added by basicConfig if file only.
    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s')) # Simpler for CLI
        logging.getLogger().addHandler(console_handler)

    logging.info("Running in Command-Line Interface (CLI) mode.")

    if not all([cli_args.szuru_url, cli_args.szuru_user, cli_args.szuru_token]):
        logging.critical("Szurubooru URL, Username, or API Token is not configured. Exiting.")
        print("Szurubooru connection details are missing. Please set them via CLI arguments or environment variables.")
        sys.exit(1)
    if not os.path.exists(cli_args.captioner_script_path):
        logging.critical(f"AI Captioner script not found at the specified path: {cli_args.captioner_script_path}")
        sys.exit(1)

    szuru_client = SzurubooruClient(cli_args.szuru_url, cli_args.szuru_user, cli_args.szuru_token)
    logging.info("Szurubooru client initialized for CLI mode.")

    all_image_files_to_process = []
    supported_extensions = tuple(ext.lower() for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'])

    for input_path_arg in cli_args.image_paths: # This is the positional arg from CLI
        if os.path.isfile(input_path_arg):
            if input_path_arg.lower().endswith(supported_extensions):
                all_image_files_to_process.append(os.path.abspath(input_path_arg))
            else:
                logging.warning(f"Skipping non-supported file: {input_path_arg}")
        elif os.path.isdir(input_path_arg):
            if cli_args.recursive:
                for root_dir, _, files in os.walk(input_path_arg):
                    for file in files:
                        if file.lower().endswith(supported_extensions):
                            all_image_files_to_process.append(os.path.join(root_dir, file))
            else:
                for file in os.listdir(input_path_arg):
                    if file.lower().endswith(supported_extensions):
                         all_image_files_to_process.append(os.path.join(input_path_arg, file))
        else:
            logging.warning(f"Input path '{input_path_arg}' is not a valid file or directory. Skipping.")

    if not all_image_files_to_process:
        logging.info("No image files found to process in CLI mode.")
        sys.exit(0)

    logging.info(f"Found {len(all_image_files_to_process)} image(s) to process via CLI.")
    successful_processing_count = 0
    failed_processing_count = 0

    for i, image_file in enumerate(all_image_files_to_process):
        logging.info(f"\nCLI Processing image {i+1}/{len(all_image_files_to_process)}: {os.path.basename(image_file)}")
        if process_single_image(image_file, szuru_client, cli_args): # Pass cli_args directly
            successful_processing_count += 1
        else:
            failed_processing_count += 1

    logging.info(f"CLI Processing finished. Success: {successful_processing_count}, Failed: {failed_processing_count}")


# --- PASTE THE CORE FUNCTIONS HERE ---
# Make sure to paste the following functions into this script:
# - parse_captioner_output(raw_output_str: str) -> dict
# - run_captioning_model(...) -> dict | None
# - class SzurubooruClient: ... (with all its methods)
# - calculate_md5(file_path)
# - process_single_image(...) -> bool
# (These were defined in the previous version of szuru_uploader.py)
# For example:
def parse_captioner_output(raw_output_str: str) -> dict:
    logging.debug("Attempting to parse captioner output.")
    parsed_data = {
        "description": "", "details": "", "style": "",
        "query": "", "tags": [], "rating": "general"
    }
    sections = ["DESCRIPTION", "DETAILS", "STYLE", "QUERY", "TAGS", "RATING"]
    current_text_block = raw_output_str
    for section_name in sections:
        try:
            start_tag = f"<{section_name}>\n"
            end_tag = f"\n</{section_name}>"
            if start_tag in current_text_block:
                content_plus_remainder = current_text_block.split(start_tag, 1)[1]
                if end_tag in content_plus_remainder:
                    content = content_plus_remainder.split(end_tag, 1)[0]
                    current_text_block = content_plus_remainder.split(end_tag, 1)[1]
                else:
                    simple_end_tag = f"</{section_name}>"
                    if content_plus_remainder.strip().endswith(simple_end_tag):
                        content = content_plus_remainder.strip()[:-len(simple_end_tag)].strip()
                    else:
                        content = content_plus_remainder.strip()
                    current_text_block = ""
                if section_name == "TAGS":
                    parsed_data["tags"] = [t.strip() for t in content.split(',') if t.strip()]
                else:
                    parsed_data[section_name.lower()] = content.strip()
            else:
                logging.warning(f"Section <{section_name}> not found in captioner output.")
        except Exception as e:
            logging.error(f"Error parsing section {section_name}: {e}. Content block: '{current_text_block[:100]}'")
    if not any(parsed_data.values()):
        logging.error("Parsing resulted in empty data.")
        return {}
    logging.debug(f"Parsed AI data: {parsed_data}")
    return parsed_data

def run_captioning_model(image_path: str, py_executable: str, script_path: str, device: str, dtype: str, max_tokens: int) -> dict | None:
    logging.info(f"Requesting AI captions for: {image_path}")
    if not os.path.exists(script_path):
        logging.error(f"Captioning script not found at: {script_path}")
        return None
    if not os.path.exists(image_path):
        logging.error(f"Image file not found for captioning: {image_path}")
        return None
    command = [py_executable, script_path, image_path, "--device", device, "--dtype", dtype, "--max_tokens", str(max_tokens)]
    logging.debug(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True, timeout=600)
        raw_output_stdout = process.stdout.strip()
        logging.debug(f"Raw STDOUT from captioner for {image_path}:\n{raw_output_stdout}")
        if process.stderr: logging.debug(f"Raw STDERR from captioner for {image_path}:\n{process.stderr.strip()}")
        if not raw_output_stdout:
            logging.error(f"Captioning script produced no output to STDOUT for {image_path}.")
            return None
        caption_block_start_tag = "<DESCRIPTION>"
        actual_caption_block_index = raw_output_stdout.find(caption_block_start_tag)
        if actual_caption_block_index != -1:
            caption_block_to_parse = raw_output_stdout[actual_caption_block_index:]
        else:
            logging.warning(f"Could not find '{caption_block_start_tag}'. Parsing full output: {raw_output_stdout[:500]}")
            caption_block_to_parse = raw_output_stdout
        parsed_data = parse_captioner_output(caption_block_to_parse)
        if not parsed_data or (not parsed_data.get("tags") and not parsed_data.get("description")):
            logging.error(f"Failed to parse essential AI data for {image_path}.")
            return None
        logging.info(f"AI captioning successful for {image_path}")
        return parsed_data
    except subprocess.CalledProcessError as e:
        logging.error(f"Captioner script error (RC: {e.returncode}) for {image_path}:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logging.error(f"Captioner script timed out for {image_path}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error running captioning for {image_path}: {e}")
        return None

class SzurubooruClient:
    def __init__(self, base_url, username, api_token):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({"Accept": "application/json"})
        self.last_status_code = None # To help with 204 debugging

    def _api_request(self, method, endpoint, **kwargs):
        url = f"{self.base_url}{endpoint}"
        self.last_status_code = None # Reset
        if 'files' not in kwargs and ('data' not in kwargs and 'json' not in kwargs) and method in ['POST', 'PUT', 'PATCH']:
            if 'body' in kwargs: kwargs['json'] = kwargs.pop('body')
        logging.debug(f"Szurubooru API Req: {method} {url} Auth: {self.auth[0]}/**** Data: {kwargs.get('json', kwargs.get('data')) is not None} Files: {kwargs.get('files') is not None}")
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            self.last_status_code = response.status_code
            logging.debug(f"Szurubooru API Resp Status: {response.status_code}")
            if response.text: logging.debug(f"Szurubooru API Resp Content: {response.text[:500]}")
            response.raise_for_status()
            if response.status_code == 204: return True
            if response.content: return response.json()
            return None # Should not happen for 200/201 if content is expected
        except requests.exceptions.HTTPError as e:
            logging.error(f"Szurubooru API HTTP error: {e.response.status_code} - {e.response.text} for {method} {url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Szurubooru API request failed for {method} {url}: {e}")
        return None

    def map_ai_rating_to_szuru(self, ai_rating_str: str) -> str:
        ai_rating_str = ai_rating_str.lower()
        mapping = {"general": "safe", "suggestive": "sketchy", "explicit": "unsafe"}
        default_szuru_rating = "safe"
        szuru_rating = mapping.get(ai_rating_str, default_szuru_rating)
        if szuru_rating == default_szuru_rating and ai_rating_str not in mapping:
            logging.warning(f"Unknown AI rating '{ai_rating_str}', defaulting to '{default_szuru_rating}'.")
        return szuru_rating

    def upload_image(self, image_path: str, initial_tags: list = None, safety: str = "safe", description: str = None, source: str = None):
        logging.info(f"Uploading {image_path} to Szurubooru...")
        if initial_tags is None: initial_tags = []
        try:
            with open(image_path, 'rb') as f:
                files = {'content': (os.path.basename(image_path), f, 'application/octet-stream')}
                post_data = {"tags": " ".join(initial_tags), "safety": safety}
                if description: post_data["description"] = description
                if source: post_data["source"] = source
                response_data = self._api_request('POST', '/posts', files=files, data=post_data)
                if response_data and isinstance(response_data, dict) and response_data.get('id'):
                    logging.info(f"Image {image_path} uploaded. Post ID: {response_data['id']}")
                    return response_data['id']
                else:
                    logging.error(f"Failed to get post ID from Szurubooru after upload. Response: {response_data}")
                    return None
        except FileNotFoundError: logging.error(f"Image file not found for upload: {image_path}"); return None
        except Exception as e: logging.error(f"Unexpected error during image upload for {image_path}: {e}"); return None

    def update_post_metadata(self, post_id: int, tags: list = None, description: str = None, safety: str = None):
        logging.info(f"Updating metadata for Szurubooru post ID: {post_id}")
        payload = {"post": {}}
        if tags is not None: payload["post"]["tags"] = " ".join(tags)
        if description is not None: payload["post"]["description"] = description
        if safety is not None: payload["post"]["safety"] = safety
        if not payload["post"]: logging.info(f"No metadata changes for post {post_id}."); return True
        response = self._api_request('PUT', f'/posts/{post_id}', json=payload)
        if response is True or (isinstance(response, dict) and response.get('id') == post_id):
            logging.info(f"Successfully updated metadata for post ID: {post_id}")
            return True
        # Check if last status was 204 explicitly if response is None but should be True
        elif response is None and self.last_status_code == 204:
             logging.info(f"Successfully updated metadata for post ID: {post_id} (204 No Content)")
             return True
        else:
            logging.error(f"Failed to update metadata for post ID {post_id}. Response: {response}")
        return False

    def check_if_image_exists_by_hash(self, image_md5_hash: str) -> int | None:
        logging.info(f"Checking for existing image with MD5 hash: {image_md5_hash}")
        response = self._api_request('GET', '/posts', params={'query': f'md5:{image_md5_hash}'})
        if response and isinstance(response, dict) and 'results' in response:
            if response['results'] and len(response['results']) > 0:
                existing_post_id = response['results'][0]['id']
                logging.info(f"Image with MD5 hash {image_md5_hash} already exists. Post ID: {existing_post_id}")
                return existing_post_id
            else:
                logging.info(f"No image found with MD5 hash: {image_md5_hash}")
                return None
        else:
            logging.warning(f"Could not determine if image exists by hash {image_md5_hash}. API response: {response}")
            return None

def calculate_md5(file_path):
    import hashlib
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError: logging.error(f"File not found for MD5: {file_path}"); return None
    except Exception as e: logging.error(f"Error calculating MD5 for {file_path}: {e}"); return None

def process_single_image(image_path: str, szuru_client: SzurubooruClient, current_args) -> bool:
    logging.info(f"Starting full processing for image: {image_path}")
    if not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}. Skipping.")
        return False
    image_md5 = calculate_md5(image_path)
    if not image_md5: logging.error(f"Could not calculate MD5 for {image_path}. Skipping."); return False
    existing_post_id = None
    if not current_args.force_reupload:
        existing_post_id = szuru_client.check_if_image_exists_by_hash(image_md5)
    if existing_post_id and not current_args.force_update_existing:
        logging.info(f"Image {image_path} (MD5: {image_md5}) exists as post {existing_post_id}, not forcing update. Skipping.")
        return True
    ai_data = run_captioning_model(image_path, py_executable=current_args.captioner_py_exe, script_path=current_args.captioner_script_path, device=current_args.captioner_device, dtype=current_args.captioner_dtype, max_tokens=current_args.captioner_max_tokens)
    if not ai_data: logging.error(f"AI captioning failed for {image_path}."); return False
    ai_tags = ai_data.get("tags", [])
    ai_rating_str = ai_data.get("rating", "general")
    szuru_safety_rating = szuru_client.map_ai_rating_to_szuru(ai_rating_str)
    description_parts = []
    if ai_data.get("description"): description_parts.append(f"AI Description:\n{ai_data['description']}")
    if ai_data.get("details"): description_parts.append(f"AI Details:\n{ai_data['details']}")
    if ai_data.get("style"): description_parts.append(f"AI Style:\n{ai_data['style']}")
    if ai_data.get("query"): description_parts.append(f"AI Suggested Query:\n{ai_data['query']}")
    full_ai_description = "\n\n---\n\n".join(filter(None, description_parts))
    final_tags = ai_tags + [f"ai_processed_v:{current_args.ai_version_tag}"]
    if existing_post_id and current_args.force_update_existing:
        logging.info(f"Image {image_path} exists as post {existing_post_id}. Force updating metadata.")
        update_success = szuru_client.update_post_metadata(existing_post_id, tags=final_tags, description=full_ai_description, safety=szuru_safety_rating)
        if update_success: logging.info(f"Successfully updated metadata for existing post {existing_post_id}."); return True
        else: logging.error(f"Failed to update metadata for existing post {existing_post_id}."); return False
    elif not existing_post_id:
        logging.info(f"Image {image_path} (MD5: {image_md5}) does not exist. Uploading with AI data.")
        new_post_id = szuru_client.upload_image(image_path, initial_tags=final_tags, safety=szuru_safety_rating, description=full_ai_description)
        if new_post_id: logging.info(f"Successfully uploaded {image_path} as new post {new_post_id}."); return True
        else: logging.error(f"Failed to upload {image_path}."); return False
    return False # Should only be reached if existing_post_id is true but force_update is false (already handled)


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processes images using an AI captioner and uploads/updates them on Szurubooru. Runs GUI if no image_paths provided.")
    # CLI specific args
    parser.add_argument("image_paths", nargs='*', default=[], help="Path(s) to image file(s) or folder(s) to process (CLI mode).")
    # Shared args (defaults will be used by GUI if not overridden by CLI)
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively process images in subfolders.")
    parser.add_argument("--force_reupload", action="store_true", help="Force re-upload of images even if MD5 hash matches (creates duplicates).")
    parser.add_argument("--force_update_existing", action="store_true", help="If image MD5 matches, update its metadata with new AI data.")
    parser.add_argument("--ai_version_tag", type=str, default=DEFAULT_AI_VERSION_TAG, help=f"Version tag to add (e.g., ai_processed_v:1.0). Default: {DEFAULT_AI_VERSION_TAG}")

    parser.add_argument("--szuru_url", default=os.environ.get("SZURUBOORU_API_URL", SZURUBOORU_API_URL), help="Szurubooru instance URL.")
    parser.add_argument("--szuru_user", default=os.environ.get("SZURUBOORU_USERNAME", SZURUBOORU_USERNAME), help="Szurubooru API Username.")
    parser.add_argument("--szuru_token", default=os.environ.get("SZURUBOORU_API_KEY", SZURUBOORU_API_KEY), help="Szurubooru API Token/Key.")

    parser.add_argument("--captioner_py_exe", default=DEFAULT_CAPTIONER_PYTHON_EXECUTABLE, help="Python executable for captioner.")
    parser.add_argument("--captioner_script_path", default=DEFAULT_CAPTIONER_SCRIPT_PATH, help="Path to Anime-image-Captioner.py.")
    parser.add_argument("--captioner_device", default=DEFAULT_CAPTIONER_DEVICE, choices=["auto", "cpu", "cuda"], help="Device for captioner model.")
    parser.add_argument("--captioner_dtype", default=DEFAULT_CAPTIONER_DTYPE, choices=["bfloat16", "float16", "float32"], help="Torch dtype for captioner.")
    parser.add_argument("--captioner_max_tokens", type=int, default=DEFAULT_CAPTIONER_MAX_TOKENS, help="Max new tokens for captioner.")

    args = parser.parse_args()

    if args.image_paths: # If image_paths are provided, run in CLI mode
        if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers if h.stream == sys.stdout):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter('[CLI] %(levelname)s: %(message)s'))
            logging.getLogger().addHandler(console_handler)
        main_cli(args)
    elif GUI_AVAILABLE:
        logging.info("No input paths provided via CLI. Attempting to launch GUI mode...")
        print("Launching GUI mode...") # Also print to console for visibility
        root = tk.Tk()
        app = UploaderApp(root)
        root.mainloop()
    else:
        logging.error("GUI is unavailable (Tkinter might be missing or display not found) and no input_path provided for CLI mode.")
        print("ERROR: GUI unavailable and no input for CLI mode.")
        parser.print_help()