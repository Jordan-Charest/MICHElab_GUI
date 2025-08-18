import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import binary_dilation
from skimage.morphology import black_tophat, disk
import sys
import os

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hdf5 import get_data_from_dataset, save_data_to_dataset
from utils.parsing import parse_key_value_args
from utils.parsing import smart_cast

def compute_intensity_mask(filename, avg_path, method="threshold", params={"threshold": 0.09, "dilation": 1, "structuring_element": 5}):
        
    avg_data, _ = get_data_from_dataset(filename, avg_path)

    if avg_data.ndim != 2:
        raise ValueError("The data used to compute the intensity mask should be in 2d.")
    
    if method == "threshold":
        mask = avg_data < params["threshold"]
        
    elif method == "black_tophat":
        selem = disk(params["structuring_element"])
        blackhat = black_tophat(avg_data, selem)
        mask = blackhat > params["threshold"]
        
    if params["dilation"] > 0:
        mask = binary_dilation(mask, iterations=params["dilation"])

    return mask

def apply_intensity_mask(filename, dataset_path, mask):
    
    data, attributes = get_data_from_dataset(filename, dataset_path)
    if data.ndim == 2:
        data[mask] = np.nan
    elif data.ndim == 3:
        data[:, mask] = np.nan
 
    return data, attributes

def intensity_mask_to_data(filename, dataset_paths, mask_path, params, method="threshold", copy_name=None):

    mask = compute_intensity_mask(filename, mask_path, method=method, params=params)

    for dataset_path in dataset_paths:

        data, attributes = apply_intensity_mask(filename, dataset_path, mask)

        if method == "threshold":
            mask_attributes = {"threshold_mask_path": mask_path, "threshold_mask_threshold": params["threshold"], "threshold_mask_dilation": params["dilation"]}
            attributes.update(mask_attributes)
        
        if method == "black_tophat":
            mask_attributes = ({"mask_method": method, "bth_mask_path": mask_path, "bth_mask_threshold": params["threshold"], "bth_mask_dilation": params["dilation"], "bth_mask_structuring_element": params["structuring_element"]})
            attributes.update(mask_attributes)

        if copy_name is None:
            save_data_to_dataset(filename, dataset_path, data, attributes)
        else:
            save_data_to_dataset(filename, dataset_path+"_"+copy_name, data, attributes)


class MaskPreviewApp:
    def __init__(self, root, filename, dataset_list):
        self.root = root
        self.root.title("Intensity Mask Preview")
        self.filename = filename
        self.dataset_list = dataset_list
        self.mask_path = dataset_list[0]
        self.copy_name = None

        self.method_params = {
            "threshold": (("threshold", 0.08), ("dilation", 1)),
            "black_tophat": (("structuring_element", 5), ("threshold", 0.02), ("dilation", 1)),
            "None": ()
        }
        self.param_widgets = []

        tk.Label(root, text="Select Dataset").grid(row=0, column=0, padx=10, pady=5)
        self.dataset_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
        self.dataset_combobox.current(0)
        self.dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_preview)

        # --- Method and Parameters Section ---
        param_frame = ttk.Frame(self.root)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10), padx=10)
        row = 0

        ttk.Label(param_frame, text="Method:").grid(row=row, column=0, sticky="w")
        self.method_var = tk.StringVar(value="threshold")
        method_menu = ttk.OptionMenu(param_frame, self.method_var, "threshold", *self.method_params.keys(), command=self.update_params)
        method_menu.grid(row=row, column=1, sticky="w", padx=(0, 10))

        self.dynamic_param_frame = ttk.Frame(param_frame)
        self.dynamic_param_frame.grid(row=0, column=2, sticky="w")

        self.param_frame = param_frame  # Store for later dynamic use
        self.update_params("threshold")

        # ------------------------------------------

        self.make_copy_var = tk.BooleanVar()
        self.make_copy_check = tk.Checkbutton(root, text="Make Copy", variable=self.make_copy_var, command=self.toggle_copy_name)
        self.make_copy_check.grid(row=3, column=0, columnspan=2, pady=5)

        tk.Label(root, text="Copy Name").grid(row=4, column=0, padx=10, pady=5)
        self.copy_name_entry = tk.Entry(root, state="disabled")
        self.copy_name_entry.grid(row=4, column=1, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=5, padx=10, pady=5)
        self.ax.axis('off')

        tk.Button(root, text="Preview", command=self.update_preview).grid(row=5, column=0, columnspan=2, padx=10, pady=10)
        tk.Button(root, text="Compute", command=self.compute).grid(row=6, column=0, columnspan=2, padx=10, pady=10)
        
        self.update_preview()

    def update_params(self, method_name):
        # Clear existing parameter widgets
        for widget in self.dynamic_param_frame.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

        params = self.method_params.get(method_name, ())
        for idx, (label_text, default_val) in enumerate(params):
            label = ttk.Label(self.dynamic_param_frame, text=f"{label_text}:")
            entry = ttk.Entry(self.dynamic_param_frame, width=7)
            entry.insert(0, str(default_val))

            label.grid(row=0, column=idx * 2, sticky="w", padx=(0, 2))
            entry.grid(row=0, column=idx * 2 + 1, sticky="w", padx=(0, 10))
            self.param_widgets.append((label, entry))

    def update_preview(self, event=None):
        try:
            self.mask_path = self.dataset_combobox.get()
            param_dict = {label.cget("text")[:-1]: smart_cast(entry.get()) for label, entry in self.param_widgets}
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values.")
            return

        mask = compute_intensity_mask(self.filename, self.mask_path, method=self.method_var.get(), params=param_dict)
        data, _ = apply_intensity_mask(self.filename, self.mask_path, mask)

        self.ax.clear()
        self.ax.imshow(data, cmap="viridis")
        self.ax.axis('off')
        self.canvas.draw()

    def toggle_copy_name(self):
        if self.make_copy_var.get():
            self.copy_name_entry.config(state="normal")
        else:
            self.copy_name_entry.config(state="disabled")
            self.copy_name_entry.delete(0, tk.END)

    def compute(self):
        try:
            self.threshold = float(self.threshold_entry.get())
            self.dilation = int(self.dilation_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values for Threshold and Dilation.")
            return

        self.copy_name = self.copy_name_entry.get().strip() if self.make_copy_var.get() else None
        self.root.quit()

        param_dict = {label.cget("text")[:-1]: smart_cast(entry.get()) for label, entry in self.param_widgets}
        self.result = (self.mask_path, self.method_var.get(), param_dict, self.copy_name)

def main():
    if "--batch" in sys.argv:  # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        mask_path = args["mask_path"]
        dataset_paths = [dataset for dataset in datasets if dataset != mask_path]

        intensity_mask_to_data(filename, dataset_paths, mask_path, params=args, method=args["method"], copy_name=args["copy_name"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            print("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]")
            return

        hdf5_file = sys.argv[1]
        datasets = sys.argv[2:]

        root = tk.Tk()
        app = MaskPreviewApp(root, hdf5_file, datasets)
        root.mainloop()

        mask_path, method, params, copy_name = app.result
        if mask_path:
            dataset_paths = [dataset for dataset in datasets if dataset != mask_path]

        intensity_mask_to_data(hdf5_file, dataset_paths, mask_path, method=method, params=params, copy_name=copy_name)

if __name__ == "__main__":
    main()
