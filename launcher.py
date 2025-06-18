
from gui_tabs.Visualize_DimRed_GUI import visualize_DimRedGUI
from gui_tabs.Compute_DimRed_GUI import compute_DimRedGUI
from gui_tabs.Processing_GUI import processingGUI

from tkinter import ttk
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELP_TEXT_DIR = os.path.join(BASE_DIR, "help/text")
HELP_IMG_DIR = os.path.join(BASE_DIR, "help/images")

class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unified GUI Launcher")
        self.geometry("800x600")

        top_frame = tk.Frame(self)
        top_frame.pack(side="top", fill="x")

        self.help_button = tk.Button(top_frame, text="Help", command=self.show_help)
        self.help_button.pack(side="right", padx=10, pady=1)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.help_content = load_help_content()

        visualize_dimred = visualize_DimRedGUI(self.notebook)
        compute_dimred = compute_DimRedGUI(self.notebook)
        processing = processingGUI(self.notebook)

        self.notebook.add(processing, text="HDF5 processing")
        self.notebook.add(visualize_dimred, text="Preview dim. red.")
        self.notebook.add(compute_dimred, text="Compute dim. red.")

        # Track tab changes
        self.current_tab_name = self.notebook.tab(self.notebook.select(), "text")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

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

def load_help_content():

    help_content = {
            "HDF5 processing": {
                "title": "Help: HDF5 Processing",
                "image": os.path.join(HELP_IMG_DIR, "help_processing.png"),  # Replace with valid path
                "text_file": os.path.join(HELP_TEXT_DIR, "help_processing.txt")
            },
            "Preview dim. red.": {
                "title": "Help: Dimensionality Reduction Preview",
                "image": os.path.join(HELP_IMG_DIR, "help_preview.png"),
                "text_file": os.path.join(HELP_TEXT_DIR, "/help_preview.txt")
            },
            "Compute dim. red.": {
                "title": "Help: Compute Dimensionality Reduction",
                "image": os.path.join(HELP_IMG_DIR, "help_compute.png"),
                "text_file": os.path.join(HELP_TEXT_DIR, "help_compute.txt")
            }
        }
    
    return help_content

if __name__ == "__main__":
    app = Launcher()
    app.mainloop()
