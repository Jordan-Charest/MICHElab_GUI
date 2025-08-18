import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter1d
import os
import sys

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hdf5 import get_data_from_dataset, save_data_to_dataset
from utils.parsing import parse_key_value_args

def apply_time_gauss_to_data(filename, dataset_paths, sigma=6, radius=6, copy_name=None):

    for dataset_path in dataset_paths:

        data, attributes = time_gauss_filter(filename, dataset_path, sigma=sigma, radius=radius)

        attributes.update({"time_gauss_filter_sigma": sigma, "time_gauss_filter_radius": radius})

        if copy_name is None:
            save_data_to_dataset(filename, dataset_path, data, attributes)
        else:
            save_data_to_dataset(filename, dataset_path+"_"+copy_name, data, attributes)


def time_gauss_filter(filename, dataset_path, sigma=6, radius=6):

    data, attributes = get_data_from_dataset(filename, dataset_path)

    if data.ndim == 1:
        data = gaussian_filter1d(data, sigma=sigma, axis=0, radius=radius)
    elif data.ndim == 3:
        for row in range(data.shape[1]):
            for col in range(data.shape[2]):
                data[:, row, col] = gaussian_filter1d(data[:, row, col], sigma=sigma, axis=0, radius=radius)
    else:
        raise ValueError("Cannot apply filter to data that is not 1D or 3D.")
    
    return data, attributes


class TimeGaussFilterPreviewApp:
    def __init__(self, root, filename, dataset_list):
        self.root = root
        self.root.title("Time Gaussian Filter Preview")
        self.filename = filename
        self.dataset_list = dataset_list
        self.preview_path = dataset_list[0]
        self.sigma = 6.0
        self.radius = 6
        
        tk.Label(root, text="Select Dataset").grid(row=0, column=0, padx=10, pady=5)
        self.dataset_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
        self.dataset_combobox.current(0)
        self.dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_preview)

        self.create_input("Sigma", 1, self.sigma)
        self.create_input("Radius", 2, self.radius)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=4, padx=10, pady=5)

        tk.Button(root, text="Preview", command=self.update_preview).grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        tk.Button(root, text="Compute", command=self.compute).grid(row=4, column=0, columnspan=2, padx=10, pady=10)
        
        self.update_preview()

    def create_input(self, label, row, default):
        tk.Label(self.root, text=label).grid(row=row, column=0, padx=10, pady=5)
        entry = tk.Entry(self.root)
        entry.insert(tk.END, str(default))
        entry.grid(row=row, column=1, padx=10, pady=5)
        setattr(self, f"{label.lower()}_entry", entry)

    def update_preview(self, event=None):
        try:
            self.preview_path = self.dataset_combobox.get()
            self.sigma = float(self.sigma_entry.get())
            self.radius = int(self.radius_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values.")
            return
        
        original_data, _ = get_data_from_dataset(self.filename, self.preview_path)
        

        preview_data, _ = time_gauss_filter(self.filename, self.preview_path, sigma=self.sigma, radius=self.radius)
        

        if original_data.ndim == 3:
            original_data = np.nanmean(original_data, axis=(1,2))
            preview_data = np.nanmean(preview_data, axis=(1,2))
        elif original_data.ndim == 1:
            pass
        
        self.ax1.clear()
        self.ax1.plot(original_data, label="Original Data")
        self.ax1.legend()
        
        self.ax2.clear()
        self.ax2.plot(preview_data, label="Filtered Data", color="orange")
        self.ax2.legend()
        
        self.canvas.draw()

    def compute(self):
        try:
            self.sigma = float(self.sigma_entry.get())
            self.radius = int(self.radius_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid numeric values.")
            return
        
        self.root.quit()
        self.result = (self.preview_path, self.sigma, self.radius)

def main():
    if "--batch" in sys.argv:  # Batch mode
        
        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        apply_time_gauss_to_data(filename, datasets, sigma=args["sigma"], radius=args["radius"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            print("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]")
            return

        hdf5_file = sys.argv[1]
        datasets = sys.argv[2:]

        root = tk.Tk()
        app = TimeGaussFilterPreviewApp(root, hdf5_file, datasets)
        root.mainloop()

        preview_path, sigma, radius = app.result
        
        apply_time_gauss_to_data(hdf5_file, datasets, sigma=sigma, radius=radius)

if __name__ == "__main__":
    main()
