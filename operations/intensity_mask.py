import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import binary_dilation
from toolbox_jocha.hdf5 import get_data_from_dataset, save_data_to_dataset
from toolbox_jocha.parsing import parse_key_value_args
import sys

def intensity_mask_to_data(filename, dataset_paths, mask_path, threshold=0.1, dilation=0, copy_name=None):

    mask = compute_intensity_mask(filename, mask_path, threshold=threshold, dilation=dilation)

    for dataset_path in dataset_paths:

        data, attributes = apply_intensity_mask(filename, dataset_path, mask)

        attributes.update({"mask_path": mask_path, "mask_threshold": threshold, "mask_dilation": dilation})

        if copy_name is None:
            save_data_to_dataset(filename, dataset_path, data, attributes)
        else:
            save_data_to_dataset(filename, dataset_path+"_"+copy_name, data, attributes)


def compute_intensity_mask(filename, avg_path, threshold=0.1, dilation=0):
        
        avg_data, _ = get_data_from_dataset(filename, avg_path)

        if avg_data.ndim != 2:
            raise ValueError("The data used to compute the intensity mask should be in 2d.")
        
        mask = avg_data < threshold
        if dilation > 0:
            mask = binary_dilation(mask, iterations=dilation)

        return mask

def apply_intensity_mask(filename, dataset_path, mask):

    data, attributes = get_data_from_dataset(filename, dataset_path)
    if data.ndim == 2:
        data[mask] = np.nan
    elif data.ndim == 3:
        data[:, mask] = np.nan
 
    return data, attributes

class MaskPreviewApp:
    def __init__(self, root, filename, dataset_list):
        self.root = root
        self.root.title("Intensity Mask Preview")
        self.filename = filename
        self.dataset_list = dataset_list
        self.mask_path = dataset_list[0]
        self.threshold = 0.1
        self.dilation = 0
        self.copy_name = None

        tk.Label(root, text="Select Dataset").grid(row=0, column=0, padx=10, pady=5)
        self.dataset_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
        self.dataset_combobox.current(0)
        self.dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_preview)

        tk.Label(root, text="Threshold").grid(row=1, column=0, padx=10, pady=5)
        self.threshold_entry = tk.Entry(root)
        self.threshold_entry.insert(tk.END, str(self.threshold))
        self.threshold_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(root, text="Dilation").grid(row=2, column=0, padx=10, pady=5)
        self.dilation_entry = tk.Entry(root)
        self.dilation_entry.insert(tk.END, str(self.dilation))
        self.dilation_entry.grid(row=2, column=1, padx=10, pady=5)

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

    def update_preview(self, event=None):
        try:
            self.mask_path = self.dataset_combobox.get()
            self.threshold = float(self.threshold_entry.get())
            self.dilation = int(self.dilation_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values for Threshold and Dilation.")
            return

        mask = compute_intensity_mask(self.filename, self.mask_path, self.threshold, self.dilation)
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

        self.result = (self.mask_path, self.threshold, self.dilation, self.copy_name)

def main():
    if "--batch" in sys.argv:  # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        mask_path = args["mask_path"]
        dataset_paths = [dataset for dataset in datasets if dataset != mask_path]

        intensity_mask_to_data(filename, dataset_paths, mask_path, threshold=args["threshold"], dilation=args["dilation"], copy_name=args["copy_name"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            print("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]")
            return

        hdf5_file = sys.argv[1]
        datasets = sys.argv[2:]

        root = tk.Tk()
        app = MaskPreviewApp(root, hdf5_file, datasets)
        root.mainloop()

        mask_path, threshold, dilation, copy_name = app.result
        if mask_path:
            dataset_paths = [dataset for dataset in datasets if dataset != mask_path]

        intensity_mask_to_data(hdf5_file, dataset_paths, mask_path, threshold=threshold, dilation=dilation, copy_name=copy_name)

if __name__ == "__main__":
    main()
