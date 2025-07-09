
import gui_tabs.Visualize_DimRed_GUI
import gui_tabs.Compute_DimRed_GUI
import gui_tabs.Processing_GUI
import gui_tabs.Create_HDF5_dataset_GUI
import gui_tabs.dfc_GUI

from tkinter import ttk
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import os
import sys
import traceback
from pathlib import Path

import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELP_TEXT_DIR = os.path.join(BASE_DIR, "help/text")
HELP_IMG_DIR = os.path.join(BASE_DIR, "help/images")
DISABLE_ERROR_CATCHING = False # Set this to True to disable the GUI Log error catching

class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MICHElab GUI")
        self.geometry("800x600")

        top_frame = tk.Frame(self)
        top_frame.pack(side="top", fill="x")

        self.help_button = tk.Button(top_frame, text="Help", command=self.show_help)
        self.help_button.pack(side="right", padx=10, pady=1)

        # Reload code and tabs
        self.reload_button = tk.Button(top_frame, text="Reload Tabs", command=self.reload_tabs)
        self.reload_button.pack(side="right", padx=10, pady=1)

        # For logging purposes
        self.log_buffer = []

        self.log_button = tk.Button(top_frame, text="Show Log", command=self.show_log)
        self.log_button.pack(side="right", padx=10, pady=1)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.help_content = load_help_content()

        self.init_tabs()

        # Track tab changes
        self.current_tab_name = self.notebook.tab(self.notebook.select(), "text")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        if not DISABLE_ERROR_CATCHING:
            sys.stdout = self
            sys.stderr = self
            sys.excepthook = self.handle_exception

        print("App started")

    def on_tab_change(self, event):
        selected_tab = event.widget.select()
        self.current_tab_name = self.notebook.tab(selected_tab, "text")

    def show_help(self):
        info = self.help_content.get(self.current_tab_name)
        if info:
            try:
                with open(info["text_file"], "r", encoding="utf-8") as f:
                    help_text = f.read()
            except Exception as e:
                help_text = f"Error loading help text: {e}"

            self.help_popup(info["title"], info["image"], help_text)
        else:
            self.help_popup("Help", "", "No help content available for this tab.")

    def help_popup(self, title, image_path, help_text):
        popup = tk.Toplevel(self)
        popup.title(title)

        # Display image if available
        if image_path:
            try:
                img = Image.open(image_path)
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)
                img_label = tk.Label(popup, image=photo)
                img_label.image = photo  # Keep reference
                img_label.pack(padx=10, pady=10)
            except Exception as e:
                tk.Label(popup, text=f"Could not load image: {e}").pack(padx=10, pady=10)

        # Display help text
        text_widget = ScrolledText(popup, wrap='word', width=60, height=15)
        text_widget.insert('1.0', help_text)
        text_widget.config(state='disabled')
        text_widget.pack(padx=10, pady=10)

        tk.Button(popup, text="Close", command=popup.destroy).pack(pady=(0, 10))

    def log_message(self, message):
        message = message.strip()
        if not message:
            return

        self.log_buffer.append(message)

        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.configure(state="disabled")
            self.log_text.see("end")

    def show_log(self):
        if hasattr(self, 'log_window') and self.log_window and self.log_window.winfo_exists():
            self.log_window.lift()
            return

        self.log_window = tk.Toplevel(self)
        self.log_window.title("Log Output")
        self.log_window.geometry("600x300")

        # === Top Frame for Clear Button ===
        top_button_frame = tk.Frame(self.log_window)
        top_button_frame.pack(fill="x", padx=5, pady=5)

        clear_button = tk.Button(top_button_frame, text="Clear Log", command=self.clear_log)
        clear_button.pack(side="left")

        # === Frame for log + scrollbar ===
        frame = tk.Frame(self.log_window)
        frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(frame, wrap="word", state="disabled", background="#f0f0f0")
        scrollbar = tk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === Fill from buffer ===
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for line in self.log_buffer:
            self.log_text.insert("end", line + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def clear_log(self):
        self.log_buffer = []
        if self.log_text and self.log_text.winfo_exists():
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")

    def write(self, message):
        self.log_message(message)

    def flush(self):
        pass  # Required for compatibility

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        traceback_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print("Uncaught exception:\n" + traceback_text)

    def reload_tabs(self):
        print("Reloading tabs...")

        # Step 1: Reload source modules
        importlib.reload(gui_tabs.Create_HDF5_dataset_GUI)
        importlib.reload(gui_tabs.Processing_GUI)
        importlib.reload(gui_tabs.Visualize_DimRed_GUI)
        importlib.reload(gui_tabs.Compute_DimRed_GUI)
        importlib.reload(gui_tabs.dfc_GUI)

        self.init_tabs()

    def init_tabs(self):
        # Clear existing tabs if reloading
        for tab in self.notebook.winfo_children():
            tab.destroy()

        # IMPORTANT: don't forget to update the reload_tabs method as well!
        visualize_dimred = gui_tabs.Visualize_DimRed_GUI.visualize_DimRedGUI(self.notebook)
        compute_dimred = gui_tabs.Compute_DimRed_GUI.compute_DimRedGUI(self.notebook)
        processing = gui_tabs.Processing_GUI.processingGUI(self.notebook)
        create_HDF5 = gui_tabs.Create_HDF5_dataset_GUI.create_HDF5_dataset_GUI(self.notebook)
        dfc = gui_tabs.dfc_GUI.dfcGUI(self.notebook)

        self.notebook.add(create_HDF5, text="Create HDF5")
        self.notebook.add(processing, text="HDF5 processing")
        self.notebook.add(visualize_dimred, text="Preview dim. red.")
        self.notebook.add(compute_dimred, text="Compute dim. red.")
        self.notebook.add(dfc, text="Compute dFC")

def load_help_content():

    help_content = {
        "Create HDF5": {
                "title": "Help: Create HDF5 File",
                "image": os.path.join(HELP_IMG_DIR, "help_create.png"),
                "text_file": os.path.join(HELP_TEXT_DIR, "help_create.txt")
            },
            "HDF5 processing": {
                "title": "Help: HDF5 Processing",
                "image": os.path.join(HELP_IMG_DIR, "help_processing.png"),
                "text_file": os.path.join(HELP_TEXT_DIR, "help_processing.txt")
            },
            "Preview dim. red.": {
                "title": "Help: Dimensionality Reduction Preview",
                "image": os.path.join(HELP_IMG_DIR, "help_preview_dimred.png"),
                "text_file": os.path.join(HELP_TEXT_DIR, "help_preview_dimred.txt")
            },
            "Compute dim. red.": {
                "title": "Help: Compute Dimensionality Reduction",
                "image": os.path.join(HELP_IMG_DIR, "help_compute_dimred.png"),
                "text_file": os.path.join(HELP_TEXT_DIR, "help_compute_dimred.txt")
            },
            "Compute dFC": {
                "title": "Help: Compute dynamic Functional Connectivity (dFC)",
                "image": os.path.join(HELP_IMG_DIR, "help_compute_dfc.png"),
                "text_file": os.path.join(HELP_TEXT_DIR, "help_compute_dfc.txt")
            }
        }
    
    return help_content

if __name__ == "__main__":
    app = Launcher()
    app.mainloop()
