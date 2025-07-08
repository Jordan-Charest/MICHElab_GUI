import ast
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import shutil
import subprocess

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numba as nb
import numpy as np
import tifffile
from pathlib import Path

from utils.hdf5 import get_data_from_dataset, save_dict_to_hdf5, add_attributes_to_dataset, create_hdf5
from utils.connectivity import bin_3d_matrix
from utils.ets import format_array
from numba import njit

# VALID_EXTENSIONS = [("Numpy or TIFF", "*.npy *.tif *.tiff")] # For adding datasets to existing HDF5 file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# SCRIPT_DIRECTORY = PROJECT_ROOT / "operations"


class dfcGUI(ttk.Frame):
    
    def __init__(self, parent):
        super().__init__(parent)
        self.preview_canvas = None
        self.create_widgets()

    def create_widgets(self):
        # FILE SELECTOR
        ttk.Label(self, text="Input File:").pack()
        input_frame = ttk.Frame(self)
        input_frame.pack()
        self.input_entry = ttk.Entry(input_frame, width=50)
        self.input_entry.pack(side=tk.LEFT)
        ttk.Button(input_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT)

        ttk.Label(self, text="Output File:").pack()
        output_frame = ttk.Frame(self)
        output_frame.pack()
        self.output_entry = ttk.Entry(output_frame, width=50)
        self.output_entry.pack(side=tk.LEFT)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).pack(side=tk.RIGHT)


        # DATASET EXPLORATION TABLE
        dataset_frame = ttk.Frame(self)
        dataset_frame.pack()
        self.dataset_tree = ttk.Treeview(dataset_frame, columns=("Shape", "Type"), show="tree headings", height=5, selectmode="browse")
        self.dataset_tree.heading("#0", text="Dataset Path")
        self.dataset_tree.heading("Shape", text="Shape")
        self.dataset_tree.heading("Type", text="Type")
        self.dataset_tree.pack(side=tk.LEFT)
        self.dataset_tree.bind("<<TreeviewSelect>>", self.update_attributes_display)

        # ATTRIBUTES DISPLAY
        self.attributes_text = tk.Text(dataset_frame, width=40, height=10, state=tk.DISABLED)
        self.attributes_text.pack(side=tk.RIGHT)

        # BINNING AND PREVIEW SECTION (now using grid)
        bin_preview_frame = ttk.Frame(self)
        bin_preview_frame.pack(pady=10)

        left_frame = ttk.Frame(bin_preview_frame)
        left_frame.grid(row=0, column=0, padx=10, sticky="n")

        ttk.Label(left_frame, text="Spatial bin").pack(anchor="w", pady=(0, 5))

        xy_frame = ttk.Frame(left_frame)
        xy_frame.pack(anchor="w", pady=5)
        ttk.Label(xy_frame, text="X:").pack(side=tk.LEFT)
        self.x_spinbox = tk.Spinbox(xy_frame, from_=1, to=100, width=5)
        self.x_spinbox.pack(side=tk.LEFT)
        self.x_spinbox.delete(0, tk.END)
        self.x_spinbox.insert(0, "5")

        ttk.Label(xy_frame, text="  Y:").pack(side=tk.LEFT)
        self.y_spinbox = tk.Spinbox(xy_frame, from_=1, to=100, width=5)
        self.y_spinbox.pack(side=tk.LEFT)
        self.y_spinbox.delete(0, tk.END)
        self.y_spinbox.insert(0, "5")

        ttk.Button(left_frame, text="Preview Binning", command=self.preview_binning).pack(pady=5)
        ttk.Button(left_frame, text="Compute dFC", command=self.compute_dfc).pack(pady=5)

        # PREVIEW PLOT
        self.figure = plt.Figure(figsize=(3, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.preview_canvas = FigureCanvasTkAgg(self.figure, master=bin_preview_frame)
        self.preview_canvas.get_tk_widget().grid(row=0, column=1, padx=20, sticky="n")

        # FILE SIZE ESTIMATE
        self.file_size_label = ttk.Label(self, text="Estimated file size: xxxxxx Mb")
        self.file_size_label.pack(pady=5)

    def browse_file(self, filename=None):
        if filename is None:
            self.filename = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5;*.hdf5")])
        else:
            self.filename = filename

        if self.filename:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.filename)

            self.dataset_structure = self.load_hdf5_structure()
            self.update_treeview()

    def browse_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            output_path = f"{folder_selected}/dfc.h5"
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, output_path)

    def load_hdf5_structure(self):
        structure = {}
        with h5py.File(self.filename, 'r') as file:
            def recursively_add(name, obj):
                if isinstance(obj, h5py.Dataset):
                    structure[name] = {"shape": obj.shape, "dtype": obj.dtype, "attrs": dict(obj.attrs)}
            file.visititems(recursively_add)
        return structure

    def update_treeview(self):
        self.dataset_tree.delete(*self.dataset_tree.get_children())
        for path, info in self.dataset_structure.items():
            self.dataset_tree.insert("", "end", iid=path, text=path, values=[info["shape"], info["dtype"]])

    def update_attributes_display(self, _=None):
        dataset_path = self.dataset_tree.selection()[0]
        if dataset_path:
            self.attributes_text.config(state=tk.NORMAL)
            self.attributes_text.delete(1.0, tk.END)
            self.attributes_text.insert(tk.END, f"Attributes for {dataset_path}:\n")
            for key, value in self.dataset_structure.get(dataset_path, {}).get("attrs", {}).items():
                self.attributes_text.insert(tk.END, f"{key}: {value}\n")
            self.attributes_text.config(state=tk.DISABLED)

    def preview_binning(self):

        filename = self.filename
        dataset_path = self.dataset_tree.selection()[0]

        try:
            x = int(self.x_spinbox.get())
            y = int(self.y_spinbox.get())
        except ValueError:
            return
        
        data, attr = get_data_from_dataset(filename, dataset_path)

        if data.ndim in (1, 2):
            messagebox.showerror("Error", "The selected data must be 3d.")
            return
        if x == 1 and y == 1:
            n_timesteps = data.shape[0]
            binned_data = data[0,:,:]

        else:
            n_timesteps = data.shape[0]
            data_frame = data[0,:,:]
            binned_data = bin_3d_matrix(np.asarray([data_frame]), (y, x))[0]

        # Update preview image
        self.ax.clear()
        self.ax.imshow(binned_data, cmap="viridis", aspect="auto")
        self.ax.set_title("Preview")
        self.preview_canvas.draw()

        # Update file size estimate
        estimated_size = self.estimate_file_size(np.asarray([binned_data]), n_timesteps)
        self.file_size_label.config(text=f"Estimated file size: {estimated_size:.2f} Mb")

    def estimate_file_size(self, data, length):
        """File size estimation in MB"""
        data = format_array(data, return_mask=False)
        one_frame_size = compute_dfc_flat(data).nbytes

        est_size = one_frame_size * length

        return est_size / 1024 / 1024  # convert bytes to MB

    def compute_dfc(self):
        print("Compute dFC function launched.")

        filename = self.filename
        dataset_path = self.dataset_tree.selection()[0]

        try:
            x = int(self.x_spinbox.get())
            y = int(self.y_spinbox.get())
        except ValueError:
            return
        
        data, attr = get_data_from_dataset(filename, dataset_path)

        if data.ndim in (1, 2):
            messagebox.showerror("Error", "The selected data must be 3d.")
            return
        if x == 1 and y == 1:
            binned_data = data
        else:
            binned_data = bin_3d_matrix(data, (y, x))

        flattened_data, elements_mask = format_array(binned_data, return_mask=True)

        dfc = compute_dfc_flat(flattened_data)

        cts = np.zeros(dfc.shape[0])
        for i in range(dfc.shape[0]):
            cts[i] = np.sqrt(np.nansum(np.square(dfc[i,:])))

        data_to_save = {"dfc": dfc, "cts": cts}
        attributes_to_save = {"binning_size": (x, y), "window_shape": binned_data.shape, "elements_mask": elements_mask}

        create_hdf5(filepath=self.output_entry.get(), overwrite=True)

        save_dict_to_hdf5(self.output_entry.get(), data_to_save)
        add_attributes_to_dataset(self.output_entry.get(), "dfc", attributes_to_save)

        print(f"Data saved to {self.output_entry.get()}.")

        del data_to_save, attributes_to_save, dfc, cts

@njit
def compute_dfc_flat(signals):
    T, N = signals.shape
    # print(f"{T}, {N}")
    triu_len = (N * (N + 1)) // 2  # Number of upper triangle elements (including diagonal)
    dFC_flat = np.zeros((T, triu_len))

    for t in range(T):
        idx = 0
        for i in range(N):
            for j in range(i, N):
                dFC_flat[t, idx] = signals[t, i] * signals[t, j]
                idx += 1

    return dFC_flat