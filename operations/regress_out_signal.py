import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import linregress
import os
import sys

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hdf5 import get_data_from_dataset, save_data_to_dataset
from utils.parsing import parse_key_value_args

def apply_regression_to_data(filename, dataset_paths, regress_signal_path, copy_name=None):

    dataset_paths.remove(regress_signal_path)

    for dataset_path in dataset_paths:

        data, attributes = regress_out_signal(filename, dataset_path, regress_signal_path)

        attributes.update({"regressed_out_signal": regress_signal_path})

        if copy_name is None:
            save_data_to_dataset(filename, dataset_path, data, attributes)
        else:
            save_data_to_dataset(filename, dataset_path+"_"+copy_name, data, attributes)


def regress_out_signal(filename, dataset_path, regressed_signal_path):

    data, attributes = get_data_from_dataset(filename, dataset_path)
    regress_data, _ = get_data_from_dataset(filename, regressed_signal_path)

    if regress_data.ndim == 3:
        regress_data = np.nanmean(regress_data, axis=(1,2))

    if data.ndim == 1:
        data = linregress(regress_data, data, alternative="two-sided") # TODO: Validate this
    elif data.ndim == 3:
        for row in range(data.shape[1]):
            for col in range(data.shape[2]):
                if np.isnan(data[0,row,col]):
                    continue
                else:
                    result = linregress(regress_data, data[:,row,col], alternative="two-sided")
                    data[:,row,col] = data[:,row,col] - result.slope * regress_data

    else:
        raise ValueError("Cannot apply linear regression to data that is not 1D or 3D.")
    
    return data, attributes


class TimeGaussFilterPreviewApp:
    def __init__(self, root, filename, dataset_list):
        self.root = root
        self.root.title("Regress out Signal Preview")
        self.filename = filename
        self.dataset_list = dataset_list
        self.preview_path = dataset_list[0]
        self.sigma = 6.0
        self.radius = 6
        
        tk.Label(root, text="Select Preview Dataset").grid(row=0, column=0, padx=10, pady=5)
        self.dataset_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
        self.dataset_combobox.current(0)
        self.dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_preview)

        tk.Label(root, text="Select Signal to regress out").grid(row=1, column=0, padx=10, pady=5)
        self.regress_combobox = ttk.Combobox(root, values=dataset_list, state="readonly")
        self.regress_combobox.current(0)
        self.regress_combobox.grid(row=1, column=1, padx=10, pady=5)
        self.regress_combobox.bind("<<ComboboxSelected>>", self.update_preview)

        # self.create_input("Sigma", 1, self.sigma)
        # self.create_input("Radius", 2, self.radius)

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
            self.regress_path = self.regress_combobox.get()

        except ValueError:
            messagebox.showerror("Invalid Input", "Invalid values for the datasets")
            return
        
        original_data, _ = get_data_from_dataset(self.filename, self.preview_path)
        regress_data, _ = get_data_from_dataset(self.filename, self.regress_path)
        

        preview_data, _ = regress_out_signal(self.filename, self.preview_path, self.regress_path)
        

        if original_data.ndim == 3:
            original_data = np.nanmean(original_data, axis=(1,2))
            preview_data = np.nanmean(preview_data, axis=(1,2))
        elif original_data.ndim == 1:
            pass
        
        self.ax1.clear()
        self.ax1.plot(original_data, label="Original Data")
        self.ax1.legend()
        
        self.ax2.clear()
        self.ax2.plot(preview_data, label="Data with signal regressed out", color="orange")
        self.ax2.legend()
        
        self.canvas.draw()

    def compute(self):
        
        self.root.quit()
        self.result = (self.preview_path, self.regress_path)

def main():
    if "--batch" in sys.argv:  # Batch mode
        
        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        apply_regression_to_data(filename, datasets, args["signal_to_regress_out"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            print("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]")
            return

        hdf5_file = sys.argv[1]
        datasets = sys.argv[2:]

        root = tk.Tk()
        app = TimeGaussFilterPreviewApp(root, hdf5_file, datasets)
        root.mainloop()

        preview_path, regress_path = app.result
        
        apply_regression_to_data(hdf5_file, datasets, regress_path)

if __name__ == "__main__":
    main()
