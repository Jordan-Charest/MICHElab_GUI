import os
import tkinter as tk
from tkinter import filedialog, StringVar
from tkinter import ttk
from utils.hdf5 import create_hdf5, add_data_to_hdf5
from utils.other import merge_dictionaries
import tifffile
import numpy as np

# Allowed file types
ALLOWED_FILETYPES = [("NumPy or Tiff files", "*.npy *.tif")]

class DatasetEntry(ttk.Frame):
    def __init__(self, parent, controller, file_path="", fps="", dataset_path="", attributes=""):
        super().__init__(parent)
        self.controller = controller
        self.vars = {
            'file': StringVar(value=file_path),
            'fps': StringVar(value=fps),
            'dataset_path': StringVar(value=dataset_path),
            'attributes': StringVar(value=attributes),
        }

        # === File Entry + Browse Button ===
        file_frame = ttk.Frame(self)
        file_entry = ttk.Entry(file_frame, textvariable=self.vars['file'], width=33)
        file_entry.pack(side="left", fill="x", expand=True)
        browse_button = ttk.Button(file_frame, text="Browse", width=7, command=self.browse_file)
        browse_button.pack(side="left", padx=(2, 0))
        file_frame.grid(row=0, column=0, padx=2, pady=1, sticky="ew")

        self.entries = {'file': file_entry}

        # === FPS Entry (narrow) ===
        fps_entry = ttk.Entry(self, textvariable=self.vars['fps'], width=8)
        fps_entry.grid(row=0, column=1, padx=2, pady=1, sticky="ew")
        self.entries['fps'] = fps_entry

        # === Dataset Path and Attributes ===
        for i, key, width in [(2, 'dataset_path', 20), (3, 'attributes', 20)]:
            entry = ttk.Entry(self, textvariable=self.vars[key], width=width)
            entry.grid(row=0, column=i, padx=2, pady=1, sticky="ew")
            self.entries[key] = entry

        # Bind highlight
        self.bind("<Button-1>", self.highlight)
        for widget in self.entries.values():
            widget.bind("<Button-1>", self.highlight)
        file_frame.bind("<Button-1>", self.highlight)
        browse_button.bind("<Button-1>", self.highlight)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=ALLOWED_FILETYPES)
        if file_path:
            self.vars['file'].set(file_path)

    def highlight(self, event=None):
        self.controller.clear_selection()
        self.configure(style="Selected.TFrame")
        self.controller.selected_entry = self

    def unhighlight(self):
        self.configure(style="TFrame")

    def get_values(self):
        return {k: v.get() for k, v in self.vars.items()}


class create_HDF5_dataset_GUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.selected_entry = None
        self.output_path = StringVar()

        style = ttk.Style()
        style.configure("Selected.TFrame", background="lightblue")

        # === Output File ===
        output_frame = ttk.Frame(self)
        output_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(output_frame, text="output file:").pack(side="left")
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=70)
        self.output_entry.pack(side="left", padx=2)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_file).pack(side="left", padx=2)

        self.overwrite_var = tk.BooleanVar()

        overwrite_checkbox = ttk.Checkbutton(output_frame, text="Overwrite", variable=self.overwrite_var)
        overwrite_checkbox.pack(side="left", padx=10)

        # === Dataset Section ===
        content_frame = ttk.Frame(self)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Label(content_frame, text="Datasets", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")

        self.dataset_area = ttk.Frame(content_frame)
        self.dataset_area.grid(row=1, column=0, sticky="nsew")

        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)

        self.setup_scrollable_dataset_area()

        # === Add / Delete Buttons ===
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=1, column=1, padx=(5, 0), sticky="n")
        ttk.Button(btn_frame, text="Add", command=self.add_entry).pack(pady=2)
        ttk.Button(btn_frame, text="Delete", command=self.delete_selected_entry).pack(pady=2)

        # === Save Button ===
        ttk.Button(self, text="Save file", command=self.save_file).pack(pady=10)

    def setup_scrollable_dataset_area(self):
        canvas = tk.Canvas(self.dataset_area, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.dataset_area, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        self.scroll_frame = ttk.Frame(canvas)
        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === Column Headers ===
        header = ttk.Frame(self.scroll_frame)
        header.grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="File", width=43).grid(row=0, column=0, padx=2, pady=1)
        ttk.Label(header, text="fps", width=8).grid(row=0, column=1, padx=2, pady=1)
        ttk.Label(header, text="Dataset path", width=20).grid(row=0, column=2, padx=2, pady=1)
        ttk.Label(header, text="Attributes (comma sep.)", width=25).grid(row=0, column=3, padx=2, pady=1)

        self.dataset_rows = []

    def browse_output_file(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_path.set(os.path.join(folder, "dataset.h5"))

    def add_entry(self):
        row = DatasetEntry(self.scroll_frame, controller=self)
        row.grid(row=len(self.dataset_rows) + 1, column=0, sticky="ew")
        self.dataset_rows.append(row)
        row.highlight()

    def delete_selected_entry(self):
        if self.selected_entry and self.selected_entry in self.dataset_rows:
            index = self.dataset_rows.index(self.selected_entry)
            self.dataset_rows.remove(self.selected_entry)
            self.selected_entry.destroy()
            self.selected_entry = None

            if self.dataset_rows:
                new_selection = self.dataset_rows[-1]
                new_selection.highlight()

    def clear_selection(self):
        for row in self.dataset_rows:
            row.unhighlight()

    def read_data(self, filename):
        """Read data in tiff or npy files
        """
            
        if filename[-4:] == ".tif":
            return tifffile.imread(filename)
        
        elif filename[-4:] == ".npy":
            return np.load(filename)
        
        raise ValueError("Could not recognize file extension.")

    def save_file(self):
        print("=== Saving Dataset ===")
        print(f"Output file: {self.output_path.get()}")

        if not self.dataset_rows:
            print("No dataset entries found.")
            return

        parsed_attrs = self.parse_all_attributes()

        # TODO: Actually define overwrite_var
        create_hdf5(self.output_path.get(), overwrite=self.overwrite_var.get())

        for i, (row, attr_dict) in enumerate(zip(self.dataset_rows, parsed_attrs), start=1):
            values = row.get_values()
            print(f"\nRow {i}:")
            print(f"  file: {values['file']}")
            print(f"  fps: {values['fps']}")
            print(f"  dataset_path: {values['dataset_path']}")
            print(f"  attributes (parsed): {attr_dict}")
            
            data = np.squeeze(self.read_data(values['file']))

            attr_dict = merge_dictionaries(attr_dict, {"fps": int(values["fps"]) if values["fps"].isdigit() else float(values["fps"]),
                                                       "filepath": str(values["file"])})
            
            group_path, dataset_name = values["dataset_path"].rsplit('/', 1)
            
            add_data_to_hdf5(self.output_path.get(), dataset_name, data, group_path, attributes=attr_dict)

            print("Done.")
            
            


    def parse_all_attributes(self):
        """Parses attributes for each row into a dictionary with inferred types."""
        parsed_list = []

        for row in self.dataset_rows:
            attr_str = row.vars['attributes'].get()
            parsed_dict = {}

            # Split by lines, then by commas within lines
            lines = attr_str.strip().splitlines()
            for line in lines:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                for part in parts:
                    if ':' in part:
                        key, val = part.split(":", 1)
                        key = key.strip()
                        val = val.strip()
                        # Type inference
                        if val.isdigit():
                            val = int(val)
                        else:
                            try:
                                val = float(val)
                            except ValueError:
                                val = str(val)
                        parsed_dict[key] = val

            parsed_list.append(parsed_dict)

        return parsed_list